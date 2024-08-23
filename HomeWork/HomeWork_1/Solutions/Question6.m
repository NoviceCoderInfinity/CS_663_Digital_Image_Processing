% Reading the both images
im1 = imread('./../Questionnare/goi1.jpg');
im1 = double(im1);

im2 = imread('./../Questionnare/goi2_downsampled.jpg');
im2 = double(im2);
% Taking the control points
for i=1:12; figure(1);
    imshow(im1 / 255); [x1(i), y1(i)] = ginput(1);
    imshow(im2 / 255); [x2(i), y2(i)] = ginput(1);
end;

P1 = [x1; y1; ones(1, length(x1))];
P2 = [x2; y2; ones(1, length(x2))];
T = P2 * pinv(P1);

disp('Transformation Matrix:');
disp(T);

[rows, cols, channels] = size(im1);

%Initially full black image for nearest neighbour interpolation
warped_image_nn = zeros(size(im1)); 

% Nearest Neighbour Interpolation
for i = 1:rows
    for j = 1:cols
        source_coords = T \ [j; i; 1];
        source_x = source_coords(1);
        source_y = source_coords(2);
        
        round_x = round(source_x);
        round_y = round(source_y);
        
        min_distance = inf;
        best_pixel = zeros(1, 1, size(im1, 3));
        
        % Check 3x3 neighborhood around the rounded coordinates
        for dx = -1:1
            for dy = -1:1
                nx = round_x + dx; ny = round_y + dy;
                if nx >= 1 && nx <= cols && ny >= 1 && ny <= rows
                    distance = sqrt((source_x - nx)^2 + (source_y - ny)^2);
                    if distance < min_distance
                        min_distance = distance;
                        best_pixel = im1(ny, nx, :);
                    end
                end
            end
        end
        warped_image_nn(i, j, :) = best_pixel;
    end
end



% Initialize the warped image for bilinear interpolation
warped_image_bilinear = zeros(size(im1));

for row = 1:rows
    for col = 1:cols
        source_coords = T \ [col; row; 1];
        source_x = source_coords(1); source_y = source_coords(2);
        
        lower_x = floor(source_x); upper_x = ceil(source_x);
        lower_y = floor(source_y); upper_y = ceil(source_y);
        
        delta_x = source_x - lower_x;
        delta_y = source_y - lower_y;
        
        for channel = 1:channels
            if lower_x >= 1 && upper_x <= cols && lower_y >= 1 && upper_y <= rows
                warped_image_bilinear(row, col, channel) = (1 - delta_x) * (1 - delta_y) * im1(lower_y, lower_x, channel) + ...
                                                           delta_x * (1 - delta_y) * im1(lower_y, upper_x, channel) + ...
                                                           (1 - delta_x) * delta_y * im1(upper_y, lower_x, channel) + ...
                                                           delta_x * delta_y * im1(upper_y, upper_x, channel);
            end
        end
    end
end

im2_uint8 = uint8(im2);
warped_image_nn_uint8 = uint8(warped_image_nn);
warped_image_bilinear_uint8 = uint8(warped_image_bilinear);

figure;
subplot(1, 3, 1); 
imshow(im2_uint8);
title('Reference Image (im2)');

subplot(1, 3, 2); 
imshow(warped_image_nn_uint8);
title('Warped Image (Nearest Neighbor)');

subplot(1, 3, 3); 
imshow(warped_image_bilinear_uint8);
title('Warped Image (Bilinear Interpolation)');
set(gcf, 'Position', [100, 100, 1500, 500]);