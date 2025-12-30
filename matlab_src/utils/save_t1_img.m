
function [] = save_t1_img(t1map, save_path, img_title, t1_range, mask)
if nargin < 3
    img_title = '';
end
close all
% Load the matlab file containing an example T1 map
if nargin < 4
    t1_range = [0 1000];
end
if nargin < 5
    mask = ones(size(t1map)); 
end
t1min = t1_range(1);
t1max = t1_range(2);

t1map = mask .* t1map; 
% Display the image using MATLAB
figure;
[imClip, cmap] = plot_t1(t1map, t1min, t1max);
title(img_title);
saveas(gcf, save_path);
close all

% save without colorbar
img = uint8((imClip - t1min) / (t1max-t1min) * 255);
imwrite(img, cmap, strrep(save_path, '.png','.tiff'));
end



function [imClip, rgb_vec] = plot_t1(t2map, loLev, upLev)
[imClip, rgb_vec] = relaxationColorMap('T1', t2map, loLev, upLev);
% Display the image using MATLAB
imshow(imClip, 'DisplayRange', [loLev, upLev], 'InitialMagnification', 'fit');
colormap(rgb_vec); 
cb = colorbar;
% cb.FontSize = 30; 
end