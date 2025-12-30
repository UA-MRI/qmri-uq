
function img = crop_image(img, fov)
[nx, ny] = size(img, 1:2);
nnx = fov(1); nny = fov(2);

if nx < nnx
    xpad = round((nnx - nx)/2);
    img = padarray(img, [xpad, 0, 0, 0]);
    xrange = 1:nnx;
else
    xrange = round(nx / 2 - nnx / 2 + 1):round(nx / 2 + nnx / 2);
end
if ny < nny
    ypad = round((nny - ny)/2);
    img = padarray(img, [0, ypad, 0, 0]);
    yrange = 1:nny;
else
    yrange = round(ny / 2 - nny / 2 + 1):round(ny / 2 + nny / 2);
end
img = img(xrange, yrange, :, :, :);
end