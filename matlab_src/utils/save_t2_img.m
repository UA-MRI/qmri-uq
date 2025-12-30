
function [] = save_t2_img(t2map, save_path, img_title, t2_range, mask)
if nargin < 3
    img_title = '';
end
close all
% Load the matlab file containing an example T2 map
if nargin < 4
    t2_range = [0 120];
end
if nargin < 5
    mask = ones(size(t2map)); 
end
t2min = t2_range(1);
t2max = t2_range(2);

t2map = mask .* t2map; 
% Display the image using MATLAB
figure;
[imClip, cmap] = plot_t2(t2map, t2min, t2max);
title(img_title);
saveas(gcf, save_path);
close all

% save without colorbar
img = uint8((imClip - t2min) / (t2max-t2min) * 255);
imwrite(img, cmap, strrep(save_path, '.png','.tiff'));
end



function [imClip, rgb_vec] = plot_t2(t2map, loLev, upLev)
[imClip, rgb_vec] = relaxationColorMap('T2', t2map, loLev, upLev);
% Display the image using MATLAB
imshow(imClip, 'DisplayRange', [loLev, upLev], 'InitialMagnification', 'fit');
colormap(rgb_vec); 
cb = colorbar;
% cb.FontSize = 30; 
end