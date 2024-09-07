% Loading the images
barbara = imread('barbara256.png');
kodak = imread('kodak24.png');

% Converting to double
barbara = double(barbara);
kodak = double(kodak);

% Adding Gaussian noise with mean=0 and sigma = 5
% randn generates random numbers from a standard normal distribution, which has a mean of 0 and a standard deviation of 1.
% size(kodak) returns a two-element vector representing the dimensions (number of rows and columns) of the kodak image.
sigma1 = 5;
barbara_noisy_5 = barbara + sigma1 * randn(size(barbara));
kodak_noisy_5 = kodak + sigma1 * randn(size(kodak));
% randn(size(kodak)) produces a matrix of the same size as the kodak image, where each element is a random value drawn from a normal distribution
% with a mean of 0 and a standard deviation of 1.

% Adding Gaussian noise with sigma = 10
sigma2 = 10;
barbara_noisy_10 = barbara + sigma2 * randn(size(barbara));
kodak_noisy_10 = kodak + sigma2 * randn(size(kodak));

% Saving the noisy images for reference
imwrite(uint8(barbara_noisy_5), 'barbara256_noisy_5.png');
imwrite(uint8(kodak_noisy_5), 'kodak24_noisy_5.png');
imwrite(uint8(barbara_noisy_10), 'barbara256_noisy_10.png');
imwrite(uint8(kodak_noisy_10), 'kodak24_noisy_10.png');


% Parameters
params = [2, 2; 0.1, 0.1; 3, 15];

% Apply bilateral filter for each parameter to barbara_noisy_5 and kodak_noisy_5
for i = 1:size(params, 1)
    sigma_s = params(i, 1);
    sigma_r = params(i, 2);
    
    barbara_filtered = mybilateralfilter(barbara_noisy_5, sigma_s, sigma_r);
    kodak_filtered = mybilateralfilter(kodak_noisy_5, sigma_s, sigma_r);
    
    imwrite(uint8(barbara_filtered), sprintf('barbara256_filtered_5_%d.png',i));
    imwrite(uint8(kodak_filtered), sprintf('kodak24_filtered_5_%d.png',i));
end

% Apply bilateral filter to barbara_noisy_10 and kodak_noisy_10
for i = 1:size(params, 1)
    sigma_s = params(i, 1);
    sigma_r = params(i, 2);
    
    barbara_filtered = mybilateralfilter(barbara_noisy_10, sigma_s, sigma_r);
    kodak_filtered = mybilateralfilter(kodak_noisy_10, sigma_s, sigma_r);
    
    imwrite(uint8(barbara_filtered), sprintf('barbara256_filtered_10_%d.png', i));
    imwrite(uint8(kodak_filtered), sprintf('kodak24_filtered_10_%d.png', i));
end