clear; clc;

image_dir = '../../../Questionnare/ORL'; % Directory containing the ORL dataset (PGM format images)

% Initialize parameters
N_train = 32 * 6;    % Number of training images (32 subjects, 6 images each)
N_test = 32 * 4;     % Number of testing images (32 subjects, 4 images each)
img_dim = 112 * 92;  % Each image has a resolution of 112x92 pixels
data = zeros(img_dim, N_train);  % Matrix to store all training images as column vectors
person_id = zeros(N_train, 1);   % Vector to store the identity (person number) of each training image
iter = 0;  % Counter for image loading

% Load training images from the dataset
for i = 1:32  % Loop over each subject (person)
    for j = 1:6  % Loop over each image (6 images per person for training)
        img_path = fullfile(image_dir, ['s' num2str(i)], [num2str(j) '.pgm']); % Construct image file path
        iter = iter + 1;  % Increment image counter
        data(:, iter) = double(reshape(imread(img_path), [], 1)); % Read and reshape the image to a column vector
        person_id(iter) = i; % Store the person ID corresponding to the image
    end
end

% Compute the mean face image and center the training data by subtracting the mean
mean_A = mean(data, 2);  % Compute the mean image across all training images
data = data - mean_A;  % Center the data by subtracting the mean image from each training image

% Perform Singular Value Decomposition (SVD) to find the eigenvectors of A * A'
[U, S, V] = svd(data, 'econ');  % U contains the eigenvectors (principal components), S has the singular values

% Compute eigencoefficients (projections of training images onto the eigenvectors)
eigcoeffs_training = U' * data;  % Project the centered training images onto the eigenface space

% Define a set of k values (number of eigenfaces) to evaluate recognition rates
k_values = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 170];  % Different values of k to test
rec_rates = zeros(length(k_values), 1);  % Array to store recognition rates for each value of k

% Testing phase: Loop over each test image and compute the recognition rate for each k
for k_idx = 1:length(k_values)
    k = k_values(k_idx);  % Number of eigenfaces to use for reconstruction
    rec_rate = 0;  % Variable to accumulate correct matches
    
    % Loop over all subjects and their test images (4 images per subject for testing)
    for i = 1:32  % Loop over each subject
        for j = 7:10  % Loop over test images (7th to 10th image per person)
            img_path = fullfile(image_dir, ['s' num2str(i)], [num2str(j) '.pgm']); % Construct file path for test image
            img_content = double(imread(img_path));  % Read the test image
            
            % Project the test image onto the eigenface space (using the same U matrix as training)
            eigcoeffs_im = U' * (img_content(:) - mean_A);  % Compute eigencoefficients for the test image
            
            % Compute the squared Euclidean distance between the test image and all training images in the k-dimensional subspace
            diffs = eigcoeffs_training - repmat(eigcoeffs_im, 1, N_train);  % Differences between test and training coefficients
            diffs = sum(diffs(1:k, :).^2, 1);  % Sum of squared differences in the first k dimensions
            [minval, minindex] = min(diffs);  % Find the training image with the smallest distance (nearest neighbor)
            
            % Check if the closest training image has the same person ID as the test image
            rec_rate = rec_rate + (i == person_id(minindex));  % Increment correct match count if IDs match
        end
    end
    
    % Calculate the recognition rate (percentage of correctly identified test images)
    rec_rate = rec_rate / N_test;  % Normalize by the total number of test images
    rec_rates(k_idx) = 100 * rec_rate;  % Store the recognition rate as a percentage
end

% Plot the recognition rate as a function of the number of eigenfaces (k)
figure;  % Create a new figure for the plot
plot(k_values, rec_rates, '-o', 'LineWidth', 2);  % Plot recognition rates with markers at each k value
xlabel('Number of Eigenfaces (k)');  % Label for x-axis
ylabel('Recognition Rate (%)');  % Label for y-axis
title('Recognition Rate vs Number of Eigenfaces ORL Dataset using SVD');  % Title of the plot
grid on;  % Turn on the grid for better readability
