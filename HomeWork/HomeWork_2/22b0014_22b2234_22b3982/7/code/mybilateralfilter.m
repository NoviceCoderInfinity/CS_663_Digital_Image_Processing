function output = mybilateralfilter(image, sigma_s, sigma_r) % mybilateralfilter takes the image and the bilateral filter parameters as input
    [rows, cols] = size(image);
    output = zeros(size(image));
    
    % Create Gaussian spatial kernel.
    % The range -3*sigma_s to 3*sigma_s is chosen because a Gaussian function effectively becomes zero beyond three standard deviations.
    window_size = ceil(3*sigma_s);
    [X, Y] = meshgrid(((-1)*window_size):window_size, ((-1)*window_size):window_size);
    G_s = exp(-(X.^2 + Y.^2) / (2 * sigma_s^2)); %Gaussian spatial kernel, which is same for all pixels.
    
    for i = 1:rows
        for j = 1:cols
            % Extract local region by ensuring that filter doesn't go out
            % of the boundary of the image.
            i_Min = max(i-window_size, 1);
            i_Max = min(i+window_size, rows);
            j_Min = max(j-window_size, 1);
            j_Max = min(j+window_size, cols);

            local_region = image(i_Min:i_Max, j_Min:j_Max);
            
            % Calculate Gaussian range kernel
            G_r = exp(-(local_region - image(i,j)).^2 / (2 * sigma_r^2));
            
            % Calculate the combined filter F by multiplying the spatial and range kernels
            F = G_s((i_Min:i_Max)-i+window_size+1, (j_Min:j_Max)-j+window_size+1) .* G_r;

            % Compute the output pixel value as the weighted average of the local region
            output(i, j) = sum(F(:) .* local_region(:)) / sum(F(:));
        end
    end
end