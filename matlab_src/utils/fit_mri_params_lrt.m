function [maps, stats] = fit_mri_params_lrt(data, sigma, D, options)
% FIT_MRI_PARAMS_LRT - Unified Likelihood Ratio Test (LRT) solver for qMRI.
%
% Performs dictionary-based parameter mapping with uncertainty quantification
% using the Profile Likelihood Ratio Test.
%
% USAGE:
%   [maps, stats] = fit_mri_params_lrt(data, sigma, D, options)
%
% INPUTS:
%   data    - [Nx, Ny, Nt] Complex image data.
%   sigma   - [Nt x Nt] Noise covariance matrix.
%   D       - Dictionary structure:
%             D.magnetization: [Nt x N_atoms] signal templates.
%             D.lookup_table:  [N_atoms x 2] [B1, q] values.
%   options - Struct with fields:
%       .alpha          - Significance level (default 0.05).
%       .b1_mode        - 'none', 'map', or 'range'.
%       .b1_input       - B1 Map (Nx,Ny) or Range (Nx,Ny,2). 
%       .te_truncation  - boolean (default false). Enable signal truncation.
%       .te_array       - [1 x Nt] vector (Required if te_truncation=true).
%       .trunc_factor   - scalar (default 3.0). Threshold factor for truncation.
%
% OUTPUTS:
%   maps    - Struct with fields .q, .B1, .q_mle, .B1_mle.
%   stats   - Struct with fields .q_ci, .B1_ci containing [min, max] bounds.

%% 1. Default Options
if nargin < 4, options = struct(); end
if ~isfield(options, 'alpha'), options.alpha = 0.05; end
if ~isfield(options, 'b1_mode'), options.b1_mode = 'none'; end
if ~isfield(options, 'te_truncation'), options.te_truncation = false; end
if ~isfield(options, 'trunc_factor'), options.trunc_factor = 3.0; end

[nx, ny, nt] = size(data);
N_voxels = nx * ny;
Xobs = reshape(data, N_voxels, nt).'; % [Nt x N_voxels]

%% 2. Process B1 Constraints
dict_b1 = unique(D.lookup_table(:,1));
b1_limits = get_b1_limits(options, dict_b1, N_voxels);

%% 3. Prepare Outputs
maps = struct('q', nan(N_voxels,1), 'B1', nan(N_voxels,1), ...
              'q_mle', nan(N_voxels,1), 'B1_mle', nan(N_voxels,1));
stats = struct('q_ci', nan(N_voxels,2), 'B1_ci', nan(N_voxels,2));

% Chi-Squared Threshold 
% DoF = 1 if B1 is fixed (Restricted), DoF = 2 if B1 is free
is_fixed_b1 = (b1_limits(:,1) == b1_limits(:,2));
dof_map = 2 * ones(N_voxels,1);
dof_map(is_fixed_b1) = 1;
chi2_vals = chi2inv(1 - options.alpha, dof_map);

%% 4. Main Loop: Group voxels by B1 Constraint
unique_ranges = unique(b1_limits, 'rows');

for r_idx = 1:size(unique_ranges,1)
    current_min = unique_ranges(r_idx, 1);
    current_max = unique_ranges(r_idx, 2);
    
    % Find voxels in this B1 constraint group
    v_idx = find(b1_limits(:,1) == current_min & b1_limits(:,2) == current_max);
    if isempty(v_idx), continue; end
    
    % --- Slice Dictionary ---
    dict_mask = (D.lookup_table(:,1) >= current_min & D.lookup_table(:,1) <= current_max);
    D_sub = D.magnetization(:, dict_mask);
    lut_sub = D.lookup_table(dict_mask, :);
    if isempty(D_sub), continue; end
    
    % --- Initial Estimate (Cosine Similarity) ---
    X_sub = Xobs(:, v_idx);
    X_norm = X_sub ./ vecnorm(X_sub, 2, 1);
    ip = X_norm' * conj(D_sub); 
    [~, best_atom] = max(abs(ip), [], 2);
    
    % Store Initial Estimates
    q_est = lut_sub(best_atom, 2);
    maps.q(v_idx) = q_est;
    maps.B1(v_idx) = lut_sub(best_atom, 1);
    
    % --- Determine Truncation Lengths ---
    if options.te_truncation
        cutoff_times = q_est * options.trunc_factor;
        trunc_lengths = sum(options.te_array(:)' <= cutoff_times, 2);
    else
        trunc_lengths = repmat(nt, length(v_idx), 1);
    end
    
    [u_lengths, ~, group_map] = unique(trunc_lengths);
    
    % --- Inner Loop: Process by Truncation Length ---
    for len_idx = 1:length(u_lengths)
        L = u_lengths(len_idx);
        if L < 3, continue; end 
        
        final_v_idx = v_idx(group_map == len_idx);
        
        % Slice Data/Dict/Sigma to length L
        X_L = Xobs(1:L, final_v_idx);
        D_L = D_sub(1:L, :);
        Sigma_L = sigma(1:L, 1:L);
        
        % Precompute Covariance-Weighted Terms
        S_inv_D = Sigma_L \ D_L;     
        S_inv_X = Sigma_L \ X_L;     
        
        % Term Calculation for Likelihood
        % A: [1 x Nvox] - Scaled Data Variance
        A = real(sum(conj(X_L) .* S_inv_X, 1));       
        
        % B: [Natoms x Nvox] - Data-Model Correlation
        B = D_L' * S_inv_X;                           
        
        % C: [Natoms x 1] - Model Variance
        C = real(sum(conj(D_L) .* S_inv_D, 1)).';     
        
        % Calculate Profile Likelihood Residual: (A - |B|^2/C)
        % Note: Transposes used to align dimensions to [Nvox x Natoms]
        term2 = (abs(B.').^2) ./ C.';
        resid_matrix = A.' - term2;
        
        % Clamp residuals to machine epsilon to prevent log(<=0)
        resid_matrix(resid_matrix <= 0) = eps; 
        
        % Negative Log Likelihood
        nll = L * log(resid_matrix);
        nll(isinf(nll) | isnan(nll)) = realmax;
        
        % MLE & LRT Statistic
        [min_nll, mle_idx] = min(nll, [], 2);
        lrt_stat = 2 * (nll - min_nll);
        lrt_stat(lrt_stat < 0) = 0;
        
        % Thresholding
        thresholds = chi2_vals(final_v_idx);
        valid_mask = lrt_stat <= thresholds;
        
        % Store MLE Results
        maps.q_mle(final_v_idx) = lut_sub(mle_idx, 2);
        maps.B1_mle(final_v_idx) = lut_sub(mle_idx, 1);
        
        % Calculate Confidence Intervals (Min/Max of valid region)
        q_vals = repmat(lut_sub(:,2).', length(final_v_idx), 1);
        B1_vals = repmat(lut_sub(:,1).', length(final_v_idx), 1);
        
        q_vals(~valid_mask) = NaN; 
        B1_vals(~valid_mask) = NaN;
        
        stats.q_ci(final_v_idx, 1) = min(q_vals, [], 2, 'omitnan');
        stats.q_ci(final_v_idx, 2) = max(q_vals, [], 2, 'omitnan');
        stats.B1_ci(final_v_idx, 1) = min(B1_vals, [], 2, 'omitnan');
        stats.B1_ci(final_v_idx, 2) = max(B1_vals, [], 2, 'omitnan');
    end
end

% Reshape results to image dimensions
maps.q = reshape(maps.q, nx, ny);
maps.B1 = reshape(maps.B1, nx, ny);
maps.q_mle = reshape(maps.q_mle, nx, ny);
maps.B1_mle = reshape(maps.B1_mle, nx, ny);
stats.q_ci = reshape(stats.q_ci, nx, ny, 2);
stats.B1_ci = reshape(stats.B1_ci, nx, ny, 2);
end

%% Helper: B1 Limit Processor
function b1_limits = get_b1_limits(options, dict_b1, N_voxels)
    switch lower(options.b1_mode)
        case 'none'
            b1_limits = repmat([min(dict_b1), max(dict_b1)], N_voxels, 1);
        case 'map'
            b1_in = reshape(options.b1_input, [], 1);
            [~, min_idx] = min(abs(b1_in - dict_b1'), [], 2);
            b1_snapped = dict_b1(min_idx);
            b1_limits = [b1_snapped, b1_snapped];
        case 'range'
            b1_in = reshape(options.b1_input, [], 2);
            [~, idx_min] = min(abs(b1_in(:,1) - dict_b1'), [], 2);
            [~, idx_max] = min(abs(b1_in(:,2) - dict_b1'), [], 2);
            b1_limits = [dict_b1(idx_min), dict_b1(idx_max)];
    end
end