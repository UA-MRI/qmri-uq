%% run_phantom_t2.m
%  Phantom Experiment: T2 Mapping Validation
%
%  DESCRIPTION:
%  Performs T2 mapping and uncertainty quantification on phantom data.
%  It processes scans with different acceleration factors, compares them against
%  a Gold Standard SESE reference, and saves the resulting maps and plots.
%
%  SCENARIOS:
%  1. Methods: LRT vs Bayesian
%  2. B1 Information: Free fit vs Constrained (Measured B1 +/- 10%)
%  3. Acceleration: 8192 (NUFFT), 384 (LLR), 192 (LLR) views
%
%  OUTPUTS:
%  - /images: T2 Maps and Uncertainty Maps (.png/.tiff)
%  - /plots:  Correlation plots (Reference vs Estimated)

restoredefaultpath;
clear; clc; close all;

%% 1. Setup Paths
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
addpath(genpath(script_dir));

repo_root = fileparts(script_dir);
data_dir  = fullfile(repo_root, 'data', 'phantom_t2');
dict_dir  = fullfile(repo_root, 'data', 'dictionaries');

%% 2. Experiment Configuration
% Scan Definitions (Filename must match what is in /data/phantom_t2/)
scans(1).name = '8192_NUFFT'; scans(1).file = 'scan_8192_nufft.mat'; dicts(1).file = 't2_phantom_TE.mat'; 
scans(2).name = '384_LLR';    scans(2).file = 'scan_384_llr.mat'; dicts(2).file = 't2_phantom_PC.mat'; 
scans(3).name = '192_LLR';    scans(3).file = 'scan_192_llr.mat'; dicts(3).file = 't2_phantom_PC.mat'; 

% Processing Settings
alpha_lvl = 0.05;        % 95% CI
trunc_factor = 3;        % TE Truncation threshold (3 * T2)
t2_disp_range = [0 150]; % Display range for T2 maps (ms)
uq_disp_range = [0 40];  % Display range for Uncertainty maps (ms)
fov = [96 96];           % cropped fov (voxels)
use_identity_cov = false; % Set to true to force Identity Covariance (Failure mode test)

% B1 Modes to Iterate
b1_modes = {'Free', 'Range'}; 

% Output Directory Logic
if use_identity_cov
    out_dir = fullfile(repo_root, 'matlab_output', 'phantom_t2_results_identity');
    fprintf('Running in IDENTITY COVARIANCE mode. Saving to: %s\n', out_dir);
else
    out_dir = fullfile(repo_root, 'matlab_output', 'phantom_t2_results');
    fprintf('Running in STANDARD COVARIANCE mode. Saving to: %s\n', out_dir);
end

if ~isfolder(out_dir); mkdir(out_dir); end
if ~isfolder(fullfile(out_dir, 'images')); mkdir(fullfile(out_dir, 'images')); end
if ~isfolder(fullfile(out_dir, 'plots')); mkdir(fullfile(out_dir, 'plots')); end

%% 3. Load Common Resources
fprintf('Loading Reference Data and Dictionary...\n');

try
    % Load References: sese_map (T2 Ref), b1_map (Measured B1)
    load(fullfile(data_dir, 'reference_maps.mat'), 'sese_t2map', 'sese_contrast', 'tfl_b1map');
    % Load ROIs: roi_masks (Struct or 3D array of logical masks)
    load(fullfile(data_dir, 'roi_masks.mat'), 'roi_masks');
catch
    error('Missing reference/dictionary files in %s or %s', data_dir, dict_dir);
end

% crop images
sese_t2map = crop_image(sese_t2map, fov); 
sese_contrast = crop_image(sese_contrast, fov); 
tfl_b1map = crop_image(tfl_b1map, fov); 

% Create a display mask (background suppression)
display_mask = sese_contrast(:,:,1) > 0.1*max(sese_contrast(:)); % Simple threshold on reference

% Container for Plotting Statistics
plot_db = struct(); 

%% 4. Main Processing Loop
for s_idx = 1:length(scans)
    scan = scans(s_idx);
    dict = dicts(s_idx); 
    fprintf('\n--- Processing Scan: %s ---\n', scan.name);
    
    % Load Contrast Data
    try
        load(fullfile(data_dir, scan.file), 'contrast', 'header');
    catch
        warning('Skipping %s (File not found)', scan.file); continue;
    end
    % Load Dictionary: D
    try
        load(fullfile(dict_dir, dict.file), 'D');
    catch
        warning('Skipping %s (Dict not found)', dict.file); continue;
    end
    
    % Prepare Data (Scale and Estimate Noise)
    contrast_dbl = crop_image(double(contrast) * 1e0, fov); 
    
    if use_identity_cov
        % Use Identity Matrix (assume i.i.d. noise)
        sigma = eye(size(contrast_dbl, 3));
    else
        % Estimate Covariance Empirically
        sigma = estimateNoiseCovariance(contrast_dbl, 10);
    end
    
    TE_array = (1:header.etl) * header.esp;
    
    for b_idx = 1:length(b1_modes)
        mode = b1_modes{b_idx};
        use_map = strcmp(mode, 'Range');
        fprintf('   > Mode: B1 %s\n', mode);
        
        % --- Configure Options ---
        ops = struct('alpha', alpha_lvl, 'te_truncation', size(contrast, 3) == header.etl, ...
                     'te_array', TE_array, 'trunc_factor', trunc_factor);
        
        if use_map
            % Updated Logic: Use 'range' mode with +/- 10% bounds
            ops.b1_mode = 'range';
            % Create [Nx, Ny, 2] map defining the valid interval per voxel
            ops.b1_input = cat(3, 0.9 * tfl_b1map, 1.1 * tfl_b1map); 
        else
            ops.b1_mode = 'none';
        end
        
        % --- Run Solvers ---
        % 1. LRT
        t0 = tic;
        [lrt_maps, lrt_stats] = fit_mri_params_lrt(contrast_dbl, sigma, D, ops);
        lrt_uq = lrt_stats.q_ci(:,:,2) - lrt_stats.q_ci(:,:,1); % Width of CI
        fprintf('     LRT: %.2fs | ', toc(t0));
        
        % 2. Bayesian
        t0 = tic;
        [bayes_maps, bayes_stats] = fit_mri_params_bayesian(contrast_dbl, sigma, D, ops);
        bayes_uq = bayes_stats.q_ci(:,:,2) - bayes_stats.q_ci(:,:,1);
        fprintf('Bayes: %.2fs\n', toc(t0));
        
        % --- Save Images (using provided helpers) ---
        base_name = sprintf('%s_%sB1', scan.name, mode);
        img_dir = fullfile(out_dir, 'images');
        
        % LRT Maps
        save_t2_img(lrt_maps.q, fullfile(img_dir, [base_name '_LRT_T2.png']), ...
            [strrep(base_name, '_', ' ') ' LRT T2'], t2_disp_range, display_mask);
        
        save_uq_img(lrt_uq, fullfile(img_dir, [base_name '_LRT_Unc.png']), ...
            [strrep(base_name, '_', ' ') ' LRT Unc'], uq_disp_range, display_mask);
            
        % Bayesian Maps
        save_t2_img(bayes_maps.q, fullfile(img_dir, [base_name '_Bayes_T2.png']), ...
            [strrep(base_name, '_', ' ') ' Bayes T2'], t2_disp_range, display_mask);
        
        save_uq_img(bayes_uq, fullfile(img_dir, [base_name '_Bayes_Unc.png']), ...
            [strrep(base_name, '_', ' ') ' Bayes Unc'], uq_disp_range, display_mask);
         
        % --- Extract ROI Statistics ---
        roi_res = extract_roi_stats(roi_masks, sese_t2map, lrt_maps, lrt_stats, bayes_maps, bayes_stats);
        
        % Store for final plotting
        plot_db(s_idx, b_idx).scan = scan.name;
        plot_db(s_idx, b_idx).mode = mode;
        plot_db(s_idx, b_idx).data = roi_res;
    end
end

%% 5. Generate Correlation Plots
fprintf('\nGenerating Correlation Plots...\n');
fs = 14; lw = 1.5;

% Create a figure for each B1 Mode
for b_idx = 1:length(b1_modes)
    mode = b1_modes{b_idx};
    fig = figure('Name', sprintf('T2 Correlation (%s B1)', mode), 'Color', 'w', ...
                 'Position', [100, 100, 1000, 500]);
    
    % Subplots for LRT and Bayesian
    methods = {'LRT', 'Bayesian'};
    
    for m = 1:2
        subplot(1, 2, m); hold on; grid on; box on;
        method = methods{m};
        
        % Identity Line
        plot([0 150], [0 150], 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
        
        % Scan Markers
        markers = {'o', 's', '^'}; 
        colors  = lines(length(scans));
        
        for s_idx = 1:length(scans)
            if isempty(plot_db(s_idx, b_idx).scan), continue; end % Skip if scan failed
            
            res = plot_db(s_idx, b_idx).data;
            ref_t2 = res.ref_mean;
            
            if strcmp(method, 'LRT')
                est_t2 = res.lrt_mean;
                ci_low = res.lrt_ci_low;
                ci_high = res.lrt_ci_high;
            else
                est_t2 = res.bayes_mean;
                ci_low = res.bayes_ci_low;
                ci_high = res.bayes_ci_high;
            end
            
            % Error Bar Plot (Vertical errors only)
            err_neg = est_t2 - ci_low;
            err_pos = ci_high - est_t2;
            
            errorbar(ref_t2, est_t2, err_neg, err_pos, ...
                markers{s_idx}, 'Color', colors(s_idx,:), ...
                'MarkerFaceColor', colors(s_idx,:), 'LineWidth', lw, ...
                'CapSize', 8, 'DisplayName', strrep(scans(s_idx).name, '_', ' '));
        end
        
        xlabel('Reference SESE T_2 (ms)', 'FontSize', fs);
        ylabel(sprintf('%s Estimated T_2 (ms)', method), 'FontSize', fs);
        title(sprintf('%s (B1: %s)', method, mode), 'FontSize', fs);
        xlim([0 150]); ylim([0 150]);
        axis square; % Force square aspect ratio
        if m == 1, legend('Location', 'northwest', 'FontSize', 10); end
    end
    
    saveas(fig, fullfile(out_dir, 'plots', sprintf('Correlation_Plot_%sB1.png', mode)));
end

fprintf('Done. Results saved to %s\n', out_dir);