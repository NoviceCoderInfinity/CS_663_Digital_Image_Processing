%input_image object is a 2d-array.
%hence the image passed to the function must be first read using imread()
%function
function localHistogramEqualization(input_image)
    % Get the dimensions of the input image
    [rows, cols] = size(input_image);

    % Define window size
    %window_size = 7; window_size = 31; window_size = 51;
    window_size = 71;
    half_window = floor(window_size / 2);

    % Pad the image with zeros to handle borders using the inbuilt function
    % padarray
    padded_image = padarray(input_image, [half_window, half_window], 'symmetric');

    % Initialize the output image
    output_image = zeros(size(input_image));

    % Loop through each pixel in the original image
    for i = 1:rows
        for j = 1:cols
            % Extract the 7x7 neighborhood around the current pixel
            neighborhood = padded_image(i:i+window_size-1, j:j+window_size-1);

            % Compute the histogram of the neighborhood using two nested
            % for loops for each center pixel
            hist_counts = zeros(1, 256); %row vector of length 256
            for m = 1:window_size
                for n = 1:window_size
                    intensity = neighborhood(m, n);
                    %disp(intensity);
                    %since the array indexing in matlab starts from 1 and
                    %not 0, and intensity values are from 0 to 255 only, we
                    %are incrementing the intensity varaible by 1
                    hist_counts(intensity+1) = hist_counts(intensity+1) + 1;
                end
            end

            % Compute the CDF from the histogram
            %cumsum() inbuilt function is used to calculate the cumulative
            %sums. Though we can manually do this as well.
            cdf = cumsum(hist_counts) / sum(hist_counts);

            % Map the original pixel intensity to the new intensity using the CDF
            %(formula as defined in class : z = (L-1)*summation(...)
            original_intensity = input_image(i, j);
            %here, since the cdf's x-axis is from 1 to 256,
            %original_intensity is incremented by 1
            new_intensity = cdf(original_intensity+1) * 255;
            % Store the new intensity in the output image
            output_image(i, j) = new_intensity;
        end
    end

    % Convert the output image to uint8 type
    %necessary as the result of histogram equalization may create
    %fractional intensity values. Also uint8() casting naturally rounds off
    %the number to nearest integer (and not the floor value)    
    output_image = uint8(output_image);
    imwrite(output_image, 'LHE_LC2_ws71.jpg')
end