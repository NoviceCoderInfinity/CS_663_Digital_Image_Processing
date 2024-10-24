clear; % Clear the workspace

% Set the directory containing the ORL image files
image_dir = '../../../Questionnare/ORL'; 

% Initialize parameters
N_images = 36 * 6;  % Total number of images (36 subjects, 6 images each)
img_dim = 112 * 92; % Dimension of each image (112 pixels by 92 pixels)
data = zeros(img_dim, N_images);  % Matrix to hold all image data as column vectors
person_id = zeros(N_images, 1);   % Vector to store the identity (person number) for each image
iter = 0;  % Counter for image loading

% Load images into the data matrix and record person IDs
for i = 1:36  % Loop through each subject (1 to 36)
    for j = 1:6  % Loop through each image for the subject (1 to 6)
        img_path = fullfile(image_dir, ['s' num2str(i)], [num2str(j) '.pgm']); % Construct image file path
        img_data = double(imread(img_path));  % Read the image as a double array
        iter = iter + 1;  % Increment image counter
        data(:, iter) = img_data(:);  % Reshape the image to a column vector and store it
        person_id(iter) = i;  % Store the identity of the person corresponding to the image
    end
end

% Compute the mean face image
mean_A = mean(data, 2);  % Compute the mean of all images
data = data - repmat(mean_A, 1, N_images);  % Center the data by subtracting the mean

% Compute the covariance matrix L = A' * A (smaller size) and perform eigen decomposition
L = data' * data;  % Compute the covariance matrix
[V, D] = eig(L);   % Get eigenvectors (V) and eigenvalues (D) from L
V = data * V;      % Transform the eigenvectors from L to eigenvectors of C (Covariance matrix)

% Normalize eigenvectors to have unit length
for i = 1:N_images
    V(:, i) = V(:, i) / norm(V(:, i));  % Normalize each eigenvector
end

% Reverse the order of the eigenvectors and eigenvalues
V = V(:, end:-1:1);  % Sort eigenvectors in descending order of eigenvalues
D = diag(D);  % Convert eigenvalue matrix to a diagonal vector
D = D(end:-1:1);  % Sort eigenvalues in descending order

% Compute eigencoefficients (projections of training images onto the eigenfaces)
eigcoeffs_training = V' * data;  % Project the centered training images onto the eigenface space

% Define k values for reconstruction
k_values = [2, 10, 20, 50, 75, 100, 125, 150, 175];  
figure; % Create a new figure for reconstructed images

% Reconstruct the 100th face image for different values of k
for idx = 1:length(k_values)
    k = k_values(idx);  % Current value of k for reconstruction
    img_data = V(:, 1:k) * eigcoeffs_training(1:k, 100) + mean_A; % Reconstruct the 100th face
    img_data = reshape(img_data, 112, 92); % Reshape the vector back to 2D image format
    img_data = img_data - min(img_data(:)); % Normalize for display
    img_data = img_data / max(img_data(:));  % Further normalize to range [0, 1]

    % Display reconstructed images in a 3x3 grid
    subplot(3, 3, idx);
    imshow(img_data); % Show the reconstructed image
    title(sprintf('Reconstruction with k = %d', k));  % Title indicating the value of k used
end
% 
% % Plot the first 25 eigenfaces in a 5x5 grid
% figure; % Create a new figure for eigenfaces
% for i = 1:25
%     img_data = V(:, i);  % Get the i-th eigenvector (eigenface)
%     img_data = reshape(img_data, 112, 92); % Reshape the eigenvector into 2D image format
%     img_data = img_data - min(img_data(:)); % Normalize for display
%     img_data = img_data / max(img_data(:)); % Further normalize to range [0, 1]
%     
%     % Display eigenfaces in a 5x5 grid
%     subplot(5, 5, i);
%     imshow(img_data); % Show the eigenface
%     title(sprintf('Eigenface %d', i));  % Title indicating the eigenface index
% end

% Plot the first 25 eigenfaces in a 5x5 grid and print their highest eigenvalues
figure; % Create a new figure for eigenfaces
for i = 1:25
    img_data = V(:, i);  % Get the i-th eigenvector (eigenface)
    img_data = reshape(img_data, 112, 92); % Reshape the eigenvector into 2D image format
    img_data = img_data - min(img_data(:)); % Normalize for display
    img_data = img_data / max(img_data(:)); % Further normalize to range [0, 1]
    
    % Display eigenfaces in a 5x5 grid
    subplot(5, 5, i);
    imshow(img_data); % Show the eigenface
    title(sprintf('Eigenface %d', i));  % Title indicating the eigenface index

    % Print the corresponding eigenvalue for each eigenface
    fprintf('Eigenface %d has an eigenvalue of: %.4f\n', i, D(i));
end
