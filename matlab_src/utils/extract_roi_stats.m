function res = extract_roi_stats(roi_masks, ref_map, lrt_maps, lrt_stats, bayes_maps, bayes_stats)
% EXTRACT_ROI_STATS Helper to compute mean/CI stats for plotting.
%
% INPUTS:
%   roi_masks   - Struct with fields .roi1, .roi2... OR 3D array [Nx,Ny,N_rois]
%   ref_map     - [Nx, Ny] Reference parameter map
%   lrt_maps    - Struct output from fit_mri_params_lrt (contains .q)
%   lrt_stats   - Struct output from fit_mri_params_lrt (contains .q_ci)
%   bayes_maps  - Struct output from fit_mri_params_bayesian
%   bayes_stats - Struct output from fit_mri_params_bayesian
%
% OUTPUT:
%   res - Struct with fields (arrays of length N_rois):
%         .ref_mean, .lrt_mean, .lrt_ci_low, .lrt_ci_high, ...

% Handle ROI input format
if isstruct(roi_masks)
    names = fieldnames(roi_masks);
    n_rois = length(names);
    masks = cell(1, n_rois);
    for i = 1:n_rois, masks{i} = roi_masks.(names{i}); end
else
    % Assume 3D array
    n_rois = size(roi_masks, 3);
    masks = cell(1, n_rois);
    for i = 1:n_rois, masks{i} = roi_masks(:,:,i); end
end

% Initialize output arrays
res = struct();
res.ref_mean = zeros(n_rois, 1);
res.lrt_mean = zeros(n_rois, 1);
res.lrt_ci_low = zeros(n_rois, 1);
res.lrt_ci_high = zeros(n_rois, 1);
res.bayes_mean = zeros(n_rois, 1);
res.bayes_ci_low = zeros(n_rois, 1);
res.bayes_ci_high = zeros(n_rois, 1);

% Loop ROIs
for i = 1:n_rois
    m = crop_image(logical(masks{i}), size(ref_map));
    if sum(m(:)) == 0, continue; end
    
    % Reference
    vals = ref_map(m);
    res.ref_mean(i) = mean(vals, 'omitnan');
    
    % LRT
    vals = lrt_maps.q(m);
    res.lrt_mean(i) = mean(vals, 'omitnan');
    
    ci_L = lrt_stats.q_ci(:,:,1);
    ci_R = lrt_stats.q_ci(:,:,2);
    res.lrt_ci_low(i) = mean(ci_L(m), 'omitnan');
    res.lrt_ci_high(i) = mean(ci_R(m), 'omitnan');
    
    % Bayesian
    vals = bayes_maps.q(m);
    res.bayes_mean(i) = mean(vals, 'omitnan');
    
    ci_L = bayes_stats.q_ci(:,:,1);
    ci_R = bayes_stats.q_ci(:,:,2);
    res.bayes_ci_low(i) = mean(ci_L(m), 'omitnan');
    res.bayes_ci_high(i) = mean(ci_R(m), 'omitnan');
end

end