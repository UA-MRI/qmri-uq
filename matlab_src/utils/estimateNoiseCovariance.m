function sigma = estimateNoiseCovariance(data, frame_size)
% ESTIMATENOISECOVARIANCE Estimates the noise covariance matrix.
%
%   sigma = estimateNoiseCovariance(data, frame_size)
%
%   INPUTS:
%     data       - 3D matrix (rows x cols x time_points)
%     frame_size - Size of the background frame border (default: 10)
%
%   OUTPUT:
%     sigma      - Estimated covariance matrix (time_points x time_points)

if nargin < 2
    frame_size = 10; 
end

[rows, cols, n_t] = size(data);
reshapedData = reshape(data, rows * cols, n_t);

% Create mask for background (frame border)
backgroundMask = true(rows, cols);
if length(frame_size) == 1
    fs_r1 = frame_size; fs_c1 = frame_size; 
    fs_r2 = frame_size; fs_c2 = frame_size; 
elseif length(frame_size) == 2
    fs_r1 = frame_size(1); fs_c1 = frame_size(2); 
    fs_r2 = frame_size(1); fs_c2 = frame_size(2); 
elseif length(frame_size) == 4
    fs_r1 = frame_size(1); fs_r2 = frame_size(2); 
    fs_c1 = frame_size(3); fs_c2 = frame_size(4); 
end
backgroundMask((fs_r1+1):(rows-fs_r2), (fs_c1+1):(cols-fs_c2)) = false;

% Extract background voxels
backgroundVoxels = reshapedData(backgroundMask(:), :);

sigma = cov(backgroundVoxels);
sigma = regularize_covariance(sigma, 500);

end