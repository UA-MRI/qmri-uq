%% run_simulation.m
%  Numerical Simulation for Dictionary-Based MRI Uncertainty Quantification
%
%  DESCRIPTION:
%  This script reproduces the numerical simulation experiments.
%  It evaluates the performance of LRT and Bayesian methods.
%

restoredefaultpath;
clear; clc; close all;

%% 1. Setup Paths
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'utils')); 

repo_root  = fileparts(script_dir); 
data_dir   = fullfile(repo_root, 'data', 'simulation_phantom'); 
dict_dir   = fullfile(repo_root, 'data', 'dictionaries');

%% 2. Experiment Settings
rng(2025); % Set seed for reproducibility

SNR_dB = 15;                  % Target Signal-to-Noise Ratio (dB)
T2_array = 20:20:300;          % Range of Ground Truth T2 values (ms)
B1_sim   = 1.0;               % Ground Truth B1 value
N_sim    = 100;              % Number of Monte Carlo realizations per T2
alpha_lvl = 0.05;             % Significance level (e.g., 0.05 for 95% CI)

% -- Failure Mode Testing --
% Set to true to assume i.i.d. noise (Identity Covariance) during FITTING,
% while ensuring the data is still SIMULATED with realistic correlated noise.
use_identity_cov = false;     

% -- Loop Settings --
contrast_modes = {'TE', 'PC'};
b1_modes       = [true, false]; 

%% 3. Output Configuration
if use_identity_cov
    output_dir = fullfile(repo_root, 'matlab_output', 'simulation_results_identity');
    fprintf('Running in IDENTITY COVARIANCE mode (Fit with Identity, Sim with Real). Saving to: %s\n', output_dir);
else
    output_dir = fullfile(repo_root, 'matlab_output', 'simulation_results');
    fprintf('Running in STANDARD COVARIANCE mode. Saving to: %s\n', output_dir);
end

if ~isfolder(output_dir); mkdir(output_dir); end
plot_data = []; 

%% 4. Main Simulation Loop
for c_idx = 1:length(contrast_modes)
    contrast_type = contrast_modes{c_idx};
    fprintf('\n=== Processing Contrast: %s ===\n', contrast_type);

    % --- Load Data & Dictionary ---
    try
        load(fullfile(data_dir, 'header.mat'), 'header');
        load(fullfile(data_dir, 'contrast.mat'), 'contrast'); 
        load(fullfile(dict_dir, sprintf('simulation_%s.mat', contrast_type)), 'D'); 
    catch
        error('Required files missing in data directories.');
    end

    % --- Handle Space Projection ---
    if strcmp(contrast_type, 'PC')
        basis = D.u; N_t = size(basis, 2);
        [nx, ny, n_orig] = size(contrast);
        contrast_reshaped = reshape(contrast, nx*ny, n_orig);
        contrast = reshape(contrast_reshaped * basis, nx, ny, N_t);
    else
        N_t = header.etl;
    end
    dict_atoms_full = D.magnetization; 

    % --- 1. Prepare Simulation Covariance  ---
    contrast_dbl = double(contrast) * 1e4; 
    sig_norm = 1;
    
    % Always estimate the true background structure for simulation
    sigma_background = estimateNoiseCovariance(contrast_dbl, 10);

    % Scale Covariance to match target SNR
    snr_lin = 10^(SNR_dB / 10);
    cov_scale = sig_norm^2 / (snr_lin^2 * trace(sigma_background));
    
    % sigma_sim is the "Ground Truth" covariance used to generate noise
    sigma_sim = regularize_covariance(cov_scale * sigma_background, 500);
    
    % Create real-valued covariance for complex noise generation
    Sigma_w = [real(sigma_sim), -imag(sigma_sim); imag(sigma_sim), real(sigma_sim)];

    % --- 2. Prepare Fitting Covariance ---
    if use_identity_cov
        % Solver sees Identity (ignoring correlations)
        % We do NOT need to scale this. The solver's sigma^2 parameter handles the scale.
        sigma_fit = eye(N_t); 
    else
        % Solver sees the True Covariance
        sigma_fit = sigma_sim;
    end

    % Define Truncation Parameters (TE space only)
    TE_array_full = (1:header.etl) * header.esp; 
    trunc_mult = 3; 

    % --- Inner Loop: B1 Constraints ---
    for use_b1_map = b1_modes
        
        ops = struct();
        ops.alpha = alpha_lvl;
        
        if use_b1_map
            b1_str = 'ConstrainedB1';
            b1_range_fit = [0.9, 1.1]; 
            ops.b1_mode = 'range';
            
            if strcmp(contrast_type, 'TE'), base_color = 'r'; else, base_color = 'b'; end
            lbl_desc = sprintf('%s Space, Constrained B_1', contrast_type);
        else
            b1_str = 'FreeB1';
            b1_range_fit = [0.4, 1.2]; 
            ops.b1_mode = 'none';
            
            if strcmp(contrast_type, 'TE'), base_color = 'k'; else, base_color = 'm'; end 
            lbl_desc = sprintf('%s Space, Free B_1', contrast_type);
        end
        
        % Configure Truncation
        if strcmp(contrast_type, 'PC')
            ops.te_truncation = false;
        else
            ops.te_truncation = true;
            ops.te_array = TE_array_full;
            ops.trunc_factor = trunc_mult;
        end
        
        fprintf('  > %s ...\n', lbl_desc);

        % Pre-allocate Result Storage
        curr_res = struct('T2', T2_array, ...
                          'LRT_Cov', zeros(size(T2_array)), 'Bayes_Cov', zeros(size(T2_array)), ...
                          'LRT_Size', zeros(size(T2_array)), 'Bayes_Size', zeros(size(T2_array)));

        % Initialize Output CSV
        csv_name = fullfile(output_dir, sprintf('sim_results_%ddB_%s_%s.csv', SNR_dB, contrast_type, b1_str));
        fcsv = fopen(csv_name, 'w');
        fprintf(fcsv, 'T2,SNR,LRT_Cov,Bayes_Cov,LRT_Size,Bayes_Size\n');

        % --- Run Monte Carlo Simulation over T2 values ---
        for idx = 1:length(T2_array)
            t2_true = T2_array(idx);
            
            % 1. Generate Signal
            [~, atom_idx] = min(abs(D.lookup_table(:,1) - B1_sim) + abs(D.lookup_table(:,2) - t2_true));
            clean_sig = dict_atoms_full(:, atom_idx);
            clean_sig = clean_sig ./ norm(clean_sig) * sig_norm;
            
            % 2. Add Correlated Noise (Using sigma_sim)
            noise_ri = mvnrnd(zeros(1, 2 * N_t), Sigma_w, N_sim);
            complex_noise = noise_ri(:, 1:N_t) + 1j * noise_ri(:, N_t+1:end);
            noisy_signals = reshape(repmat(clean_sig.', N_sim, 1) + complex_noise, [N_sim, 1, N_t]);
            
            % Update B1 input if restricted
            if use_b1_map
                ops.b1_input = repmat(reshape(b1_range_fit, 1, 1, 2), N_sim, 1, 1);
            end
            
            % 3. Run Solvers (Using sigma_fit)
            [~, lrt_stats] = fit_mri_params_lrt(noisy_signals, sigma_fit, D, ops);
            lrt_CI = lrt_stats.q_ci;

            [~, bayes_stats] = fit_mri_params_bayesian(noisy_signals, sigma_fit, D, ops);
            bayes_CI = bayes_stats.q_ci;
            
            % 4. Calculate Statistics
            lrt_cov = mean(t2_true >= lrt_CI(:,1) & t2_true <= lrt_CI(:,2)) * 100;
            bayes_cov = mean(t2_true >= bayes_CI(:,1) & t2_true <= bayes_CI(:,2)) * 100;
            lrt_sz = mean(lrt_CI(:,2) - lrt_CI(:,1), 'omitnan');
            bayes_sz = mean(bayes_CI(:,2) - bayes_CI(:,1), 'omitnan');
            
            curr_res.LRT_Cov(idx) = lrt_cov; curr_res.Bayes_Cov(idx) = bayes_cov;
            curr_res.LRT_Size(idx) = lrt_sz; curr_res.Bayes_Size(idx) = bayes_sz;
            
            fprintf(fcsv, '%f,%d,%f,%f,%f,%f\n', t2_true, SNR_dB, lrt_cov, bayes_cov, lrt_sz, bayes_sz);
        end
        fclose(fcsv);
        
        % Save results for plotting
        entry = struct();
        entry.Label = lbl_desc;
        entry.Color = base_color;
        entry.Results = curr_res;
        plot_data = [plot_data; entry];
    end
end

%% 5. Generate Combined Plots
fprintf('\nGenerating Combined Plots...\n');
fs = 14; lw = 2; 
plot_suffix = sprintf('%ddB', SNR_dB);
if use_identity_cov, plot_suffix = [plot_suffix '_Identity']; end

% --- Figure 1: Coverage Probability ---
fig1 = figure('Name', 'Coverage', 'Color', 'w', 'Position', [100 100 900 600]);
hold on; grid on; box on;
yline(95, 'k--', 'Target (95%)', 'LineWidth', 1.5, 'HandleVisibility', 'off');
plots = []; legends = {};

for i = 1:length(plot_data)
    pd = plot_data(i);
    p1 = plot(pd.Results.T2, pd.Results.LRT_Cov, '--', 'Color', pd.Color, 'LineWidth', lw);
    p2 = plot(pd.Results.T2, pd.Results.Bayes_Cov, ':', 'Color', pd.Color, 'LineWidth', lw);
    plots = [plots, p1, p2];
    legends{end+1} = ['LRT, ' pd.Label]; legends{end+1} = ['Bayesian, ' pd.Label];
end
xlabel('Parameter Value [ms]', 'FontSize', fs); ylabel('Coverage (%)', 'FontSize', fs);
ylim([70 101]); legend(plots, legends, 'Location', 'southwest', 'FontSize', 10, 'NumColumns', 2);
saveas(fig1, fullfile(output_dir, sprintf('MASTER_coverage_%s.png', plot_suffix)));

% --- Figure 2: Interval Size ---
fig2 = figure('Name', 'Size', 'Color', 'w', 'Position', [150 150 900 600]);
hold on; grid on; box on;
plots = []; legends = {};
for i = 1:length(plot_data)
    pd = plot_data(i);
    p1 = plot(pd.Results.T2, pd.Results.LRT_Size, '--', 'Color', pd.Color, 'LineWidth', lw);
    p2 = plot(pd.Results.T2, pd.Results.Bayes_Size, ':', 'Color', pd.Color, 'LineWidth', lw);
    plots = [plots, p1, p2];
    legends{end+1} = ['LRT, ' pd.Label]; legends{end+1} = ['Bayesian, ' pd.Label];
end
xlabel('Parameter Value [ms]', 'FontSize', fs); ylabel('Interval Size [ms]', 'FontSize', fs);
legend(plots, legends, 'Location', 'northwest', 'FontSize', 10, 'NumColumns', 2);
saveas(fig2, fullfile(output_dir, sprintf('MASTER_interval_size_%s.png', plot_suffix)));

fprintf('Done. Plots saved to %s\n', output_dir);
fclose all;
