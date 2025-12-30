function [maps, stats] = fit_mri_params_bayesian(data, sigma, D, options)
% FIT_MRI_PARAMS_BAYESIAN - Unified Bayesian solver for qMRI.
%
% Calculates marginal posterior distributions and credible intervals using
% numerical integration over the dictionary grid.
%
% USAGE:
%   [maps, stats] = fit_mri_params_bayesian(data, sigma, D, options)
%
% INPUTS:
%   data    - [Nx, Ny, Nt] Complex image data
%   sigma   - [Nt x Nt] Noise covariance
%   D       - Dictionary struct (D.magnetization, D.lookup_table)
%   options - Struct with fields:
%       .alpha          - (default 0.05) CI significance level
%       .b1_mode        - 'none', 'map', or 'range'
%       .b1_input       - Map (Nx,Ny) or Range (Nx,Ny,2). 
%       .te_truncation  - boolean (default false)
%       .te_array       - [1 x Nt] vector (Required if te_truncation=true)
%       .trunc_factor   - scalar (default 3.0)
%
% OUTPUTS:
%   maps    - Struct with fields .q, .B1 (Point Estimates).
%   stats   - Struct with fields .q_ci, .q_std, .B1_std.

%% 1. Default Options
if nargin < 4, options = struct(); end
if ~isfield(options, 'alpha'), options.alpha = 0.05; end
if ~isfield(options, 'b1_mode'), options.b1_mode = 'none'; end
if ~isfield(options, 'te_truncation'), options.te_truncation = false; end
if ~isfield(options, 'trunc_factor'), options.trunc_factor = 3.0; end

[nx, ny, nt] = size(data);
N_voxels = nx * ny;
Xobs = reshape(data, N_voxels, nt).'; 

%% 2. Process B1 Constraints
dict_b1 = unique(D.lookup_table(:,1));
b1_limits = get_b1_limits(options, dict_b1, N_voxels);

%% 3. Prepare Outputs
maps = struct('q', nan(N_voxels,1), 'B1', nan(N_voxels,1));
stats = struct('q_ci', nan(N_voxels,2), 'B1_std', nan(N_voxels,1), 'q_std', nan(N_voxels,1));

%% 4. Main Loop: Group by B1 Constraint
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
    
    % Sort dictionary to ensure grid alignment for reshape operations
    [lut_sub, sort_idx] = sortrows(lut_sub, [1, 2]); % Sort by B1 then q
    D_sub = D_sub(:, sort_idx);
    
    sub_b1_grid = unique(lut_sub(:,1));
    sub_q_grid = unique(lut_sub(:,2));
    
    if isempty(D_sub), continue; end
    
    % --- Initial Estimate (Cosine Similarity) ---
    X_sub = Xobs(:, v_idx);
    X_norm = X_sub ./ vecnorm(X_sub, 2, 1);
    [~, best_atom] = max(abs(X_norm' * conj(D_sub)), [], 2);
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
    
    % --- Core Processing Loop ---
    for len_idx = 1:length(u_lengths)
        L = u_lengths(len_idx);
        if L < 3, continue; end
        
        final_v_idx = v_idx(group_map == len_idx);
        
        X_L = Xobs(1:L, final_v_idx);
        D_L = D_sub(1:L, :);
        Sigma_L = sigma(1:L, 1:L);
        
        % 1. Compute Posteriors
        [p_q, p_b1] = compute_posterior(X_L, D_L, Sigma_L, sub_b1_grid, sub_q_grid, L);
        
        % 2. Calculate Statistics using Greedy CI Expansion
        % Use the point estimate as the anchor for the interval
        current_centers = maps.q(final_v_idx);
        [lb, ub] = calc_ci_greedy(sub_q_grid, p_q, current_centers, 1 - options.alpha);
        
        % Standard Deviation (Moment matching)
        [~, ~, std_q] = calc_stats_moments(sub_q_grid, p_q);
        [~, ~, std_b1] = calc_stats_moments(sub_b1_grid, p_b1);
        
        stats.q_ci(final_v_idx, 1) = lb;
        stats.q_ci(final_v_idx, 2) = ub;
        stats.q_std(final_v_idx) = std_q;
        stats.B1_std(final_v_idx) = std_b1;
    end
end

% Reshape results to image dimensions
maps.q = reshape(maps.q, nx, ny);
maps.B1 = reshape(maps.B1, nx, ny);
stats.q_ci = reshape(stats.q_ci, nx, ny, 2);
stats.B1_std = reshape(stats.B1_std, nx, ny);
stats.q_std = reshape(stats.q_std, nx, ny);
end

%% Helper: Compute Marginal Posteriors
function [p_q, p_b1] = compute_posterior(X, D, Sigma, b1_grid, q_grid, L)
    N_grp = size(X, 2);
    
    % Precompute terms for Likelihood
    S_inv_D = Sigma \ D;
    S_inv_X = Sigma \ X;
    
    A = real(sum(conj(X) .* S_inv_X, 1));       % [1 x N_grp]
    B = D' * S_inv_X;                           % [N_atoms x N_grp]
    C = real(sum(conj(D) .* S_inv_D, 1)).';     % [N_atoms x 1]
    
    RSS = A - (abs(B).^2 ./ C);
    RSS(RSS <= 0) = eps;
    
    log_prob = (1 - L) * log(RSS) - log(C);     % [N_atoms x N_grp]
    
    % Reshape [N_atoms] -> [n_q, n_b1]
    % Assuming Dictionary is sorted [B1(slow), q(fast)]
    n_b1 = length(b1_grid);
    n_q = length(q_grid);
    
    if n_b1 * n_q ~= size(D, 2)
        % Fallback for unstructured dictionary
        p_q = ones(n_q, N_grp) / n_q;
        p_b1 = ones(n_b1, N_grp) / n_b1;
        return;
    end
    
    log_prob_grid = reshape(log_prob, n_q, n_b1, N_grp);
    
    % Normalize Probabilities (Log-Sum-Exp trick)
    max_lp = max(reshape(log_prob_grid, [], N_grp), [], 1);
    prob_grid = exp(bsxfun(@minus, log_prob_grid, reshape(max_lp, 1, 1, N_grp)));
    
    % Marginalize (Integrate)
    % p(q) = Integrate over B1 (dimension 2)
    if n_b1 > 1
        p_q_raw = trapz(b1_grid, prob_grid, 2); 
    else
        p_q_raw = squeeze(prob_grid); 
    end
    % Ensure shape [n_q, N_grp]
    p_q_raw = reshape(p_q_raw, n_q, N_grp);
    
    % p(B1) = Integrate over q (dimension 1)
    if n_q > 1
        p_b1_raw = trapz(q_grid, prob_grid, 1);
    else
        p_b1_raw = squeeze(prob_grid);
    end
    % Ensure shape [n_b1, N_grp]
    p_b1_raw = reshape(p_b1_raw, n_b1, N_grp);
    
    % Final PDF Normalization
    if n_q > 1
        norm_q = trapz(q_grid, p_q_raw, 1);
    else
        norm_q = sum(p_q_raw, 1);
    end
    p_q = bsxfun(@rdivide, p_q_raw, norm_q);
    
    if n_b1 > 1
        norm_b1 = trapz(b1_grid, p_b1_raw, 1);
    else
        norm_b1 = sum(p_b1_raw, 1);
    end
    p_b1 = bsxfun(@rdivide, p_b1_raw, norm_b1);
end

%% Helper: Greedy Credible Interval Calculation
function [lb, ub] = calc_ci_greedy(grid_vals, probs, centers, confidence)
    % Calculates symmetric credible interval by greedily expanding from the mode
    [n_grid, n_vox] = size(probs);
    lb = nan(n_vox, 1);
    ub = nan(n_vox, 1);
    
    if n_grid == 1
        lb = centers; ub = centers; return;
    end

    for v = 1:n_vox
        p_v = probs(:, v);
        
        % Anchor to the point estimate (center)
        [~, mode_idx] = min(abs(grid_vals - centers(v)));
        L = mode_idx; R = mode_idx;
        
        accumulated_prob = 0;
        
        % Greedily expand L and R until confidence reached
        while accumulated_prob < confidence && (L > 1 || R < n_grid)
            % Check Left Step
            if L > 1
                conf_L = trapz(grid_vals(L-1:R), p_v(L-1:R));
            else
                conf_L = -1;
            end
            
            % Check Right Step
            if R < n_grid
                conf_R = trapz(grid_vals(L:R+1), p_v(L:R+1));
            else
                conf_R = -1;
            end
            
            % Move in direction of higher prob mass gain
            if conf_L > conf_R
                L = L - 1; accumulated_prob = conf_L;
            else
                R = R + 1; accumulated_prob = conf_R;
            end
        end
        
        % Fallback if confidence never reached
        if accumulated_prob < confidence
            lb(v) = min(grid_vals); ub(v) = max(grid_vals);
        else
            lb(v) = grid_vals(L); ub(v) = grid_vals(R);
        end
    end
end

%% Helper: Moments Calculation
function [lb, ub, std_val] = calc_stats_moments(grid_vals, probs)
    [n_grid, n_vox] = size(probs);
    if n_grid == 1, std_val = zeros(n_vox, 1); lb=grid_vals; ub=grid_vals; return; end
    
    mean_val = trapz(grid_vals, grid_vals .* probs, 1).'; 
    var_val = trapz(grid_vals, (grid_vals - mean_val').^2 .* probs, 1).';
    std_val = sqrt(var_val);
    lb = []; ub = [];
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