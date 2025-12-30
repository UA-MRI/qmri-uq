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
    fs_x = frame_size; fs_y = frame_size; 
else
    fs_x = frame_size(1); fs_y = frame_size(2); 
end
backgroundMask((fs_x+1):(rows-fs_x), (fs_y+1):(cols-fs_y)) = false;

% Extract background voxels
backgroundVoxels = reshapedData(backgroundMask(:), :);

sigma = cov(backgroundVoxels);
sigma = regularize_covariance(sigma, 500);

end