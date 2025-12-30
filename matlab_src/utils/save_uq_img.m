
function [] = save_uq_img(uqmap, save_path, img_title, uq_range, mask)
if nargin < 3
    img_title = '';
end
close all
% Load the matlab file containing an example T1 map
if nargin < 4
    uq_range = [0 120];
end
if nargin < 5
    mask = ones(size(uqmap)); 
end
uqmin = uq_range(1);
uqmax = uq_range(2);
% Create a Viridis-like colormap manually
N = 256; % Number of colors
r = linspace(0.267, 0.999, N)';
g = linspace(0.005, 0.893, N)';
b = linspace(0.33, 0.05, N)';
viridis = [r, g, b];

uqmap = mask .* uqmap; 

% Display the image using MATLAB
figure; ax = gca; 
imshow(uqmap, [uqmin, uqmax], 'InitialMagnification', 'fit'); 
colormap(ax, viridis);
caxis([uqmin, uqmax]); 
cb = colorbar;
% cb.FontSize = 30; 
title(img_title);
saveas(gcf, save_path);
close all

% save without colorbar
img = uint8((uqmap - uqmin) / (uqmax-uqmin) * 255);
% cmap = viridis; 
imwrite(img, viridis, strrep(save_path, '.png','.tiff'));
end