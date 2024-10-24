clear; clc;

% Directory containing the ORL dataset with subfolders 's1', 's2', etc.
image_dir = '../../../Questionnare/ORL'; 

% Define the number of training and test images and the image dimension
N_train = 32 * 6;  % 32 persons, 6 images per person for training
N_test = 32 * 4;   % 32 persons, 4 images per person for testing
img_dim = 112 * 92; % Each image is 112 x 92 pixels
data = zeros(img_dim, N_train); % Matrix to hold training images as column vectors
person_id = zeros(N_train, 1);  % Vector to store the person ID (for recognition)
iter = 0;  % Counter for indexing the training images

% Load the training images and reshape them into column vectors
for i = 1:32  % Loop over the 32 persons
    for j = 1:6  % Loop over the 6 training images per person
        % Construct the path to the image file
        img_path = fullfile(image_dir, ['s' num2str(i)], [num2str(j) '.pgm']); 
        iter = iter + 1;  % Increment image counter
        % Read the image, reshape it into a column vector, and store it in 'data'
        data(:, iter) = double(reshape(imread(img_path), [], 1)); 
        % Record the person ID (1 to 32) associated with each image
        person_id(iter) = i;  
    end
end

% Compute the mean of all training images (mean face)
mean_A = mean(data, 2); 
% Subtract the mean face from all training images (center the data)
data = data - mean_A; 

% Compute the covariance matrix (in a reduced form: L = A' * A)
L = data' * data; 

% Compute the eigenvectors and eigenvalues of the reduced covariance matrix
[V, D] = eig(L); 

% Map the eigenvectors of L to eigenvectors of the larger covariance matrix (C = A * A')
V = data * V; 

% Normalize the eigenvectors to ensure they have unit norm
for i = 1:N_train
    V(:, i) = V(:, i) / norm(V(:, i)); 
end

% Reverse the order of the eigenvectors to get the ones corresponding to the largest eigenvalues
V = V(:, end:-1:1);  % Eigenvectors corresponding to largest eigenvalues are now first

% Compute eigencoefficients for the training images (project training data onto the eigenfaces)
eigcoeffs_training = V' * data; 

% Define the range of k values (number of eigenfaces) to test
k_values = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 170]; 
rec_rates = zeros(length(k_values), 1);  % Array to store recognition rates for each k

% Testing phase: Evaluate the recognition rate for each k
for k_idx = 1:length(k_values)
    k = k_values(k_idx);  % Current value of k (number of eigenfaces used for reconstruction)
    rec_rate = 0;  % Initialize recognition rate for this k
    
    % Loop over each person for testing
    for i = 1:32
        % Loop over the 4 test images for each person (images 7 to 10)
        for j = 7:10
            % Construct the path to the test image
            img_path = fullfile(image_dir, ['s' num2str(i)], [num2str(j) '.pgm']); 
            % Read the test image and convert it to a column vector
            img_content = double(imread(img_path));   
            % Subtract the mean face and project the test image onto the eigenfaces
            eigcoeffs_im = V' * (img_content(:) - mean_A); 
            
            % Compute the difference (Euclidean distance) between the test image's eigencoefficients
            % and those of all the training images, for the first k eigenfaces
            diffs = eigcoeffs_training - repmat(eigcoeffs_im, 1, N_train); 
            diffs = sum(diffs(1:k, :).^2, 1);  % Sum of squared differences
            
            % Find the nearest neighbor (the training image with the smallest difference)
            [minval, minindex] = min(diffs);  
            
            % Check if the nearest neighbor belongs to the correct person
            rec_rate = rec_rate + (i == person_id(minindex));  
        end
    end
    
    % Calculate the recognition rate (percentage of correctly identified test images)
    rec_rate = rec_rate / N_test;  % Divide by the total number of test images
    rec_rates(k_idx) = 100 * rec_rate;  % Store recognition rate as a percentage
end

% Plot the recognition rate as a function of the number of eigenfaces (k)
figure;
plot(k_values, rec_rates, '-o', 'LineWidth', 2);  % Plot k vs recognition rate with markers
xlabel('Number of Eigenfaces (k)');  % Label for x-axis
ylabel('Recognition Rate (%)');  % Label for y-axis
title('Recognition Rate vs Number of Eigenfaces on ORL Dataset');  % Plot title
grid on;  % Add gridlines for better readability
