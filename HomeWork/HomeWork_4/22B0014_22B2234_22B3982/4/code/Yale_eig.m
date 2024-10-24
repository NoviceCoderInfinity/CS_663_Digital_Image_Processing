clear; % Clear the workspace

% Set the directory containing the Cropped Yale image files
img_dir =  '../../../Questionnare/CroppedYale/';

% Initialize parameters for the dataset
N_indiv = 38;                % Number of individuals in the dataset
N_train_per_indiv = 40;      % Number of training images per individual

N_train = N_indiv * N_train_per_indiv;  % Total number of training images
img_dim = 192 * 168;         % Dimension of each image (192 pixels by 168 pixels)

data = zeros(img_dim, N_train);  % Matrix to hold training image data
person_id = zeros(N_train, 1);   % Vector to store the identity (individual number) for each image

iter = 0;  % Counter for loaded training images
for i = 1:N_indiv + 1  % Loop through each individual (1 to 38, skipping individual 14)
    if i == 14, continue; end;  % Skip individual 14
    
    % Construct directory path based on individual index
    if i < 10
        dirc = dir(sprintf('%s/yaleB0%d/*.pgm', img_dir, i));  % Directories for individuals 1-9
        currdir = sprintf('%s/yaleB0%d/', img_dir, i);
    else
        dirc = dir(sprintf('%s/yaleB%d/*.pgm', img_dir, i));    % Directories for individuals 10-38
        currdir = sprintf('%s/yaleB%d/', img_dir, i);
    end
    
    NumImages = length(dirc);  % Total images available in the directory
    
    % Ensure there are enough images for training
    if NumImages < N_train_per_indiv
        error('Not enough images for training in directory: %s', currdir);
    end
    
    % Load training images (first 40 images)
    for j = 1:N_train_per_indiv
        fname = sprintf('%s/%s', currdir, dirc(j).name);  % Construct the file name
        im = double(imread(fname));  % Read the image as a double array
        
        iter = iter + 1;  % Increment image counter
        data(:, iter) = im(:);  % Reshape image to a column vector and store it
        person_id(iter) = i;    % Record the identity of the person
    end
end

data = single(data);  % Convert data to single precision
mean_A = mean(data, 2);  % Compute the mean face image
data = data - repmat(mean_A, 1, N_train);  % Center the data by subtracting the mean

% Compute the covariance matrix L = A' * A (smaller size) and perform eigen decomposition
L = data' * data;  % Compute the covariance matrix
[V, D] = eig(L);   % Get eigenvectors (V) and eigenvalues (D) from L
V = data * V;      % Transform the eigenvectors from L to eigenvectors of C (Covariance matrix)

% Normalize eigenvectors to have unit length
for i = 1:N_train
    V(:, i) = V(:, i) / norm(V(:, i));  % Normalize each eigenvector
end

% Reverse the order of the columns of V
V = V(:, end:-1:1);  % Sort eigenvectors in descending order
eigcoeffs_training = V' * data;  % Compute eigencoefficients (projections of training images onto the eigenfaces)

fprintf('\n');

% Initialize testing images
iter = 0;  % Reset the image counter
TestImages = [];  % Initialize test images matrix
testid = [];     % Initialize vector to store identities of test images

% Load testing images (remaining images after training)
for i = 1:N_indiv + 1
    if i == 14, continue; end;  % Skip individual 14
    
    % Construct directory path based on individual index
    if i < 10
        dirc = dir(sprintf('%s/yaleB0%d/*.pgm', img_dir, i));  % Directories for individuals 1-9
        currdir = sprintf('%s/yaleB0%d/', img_dir, i);
    else
        dirc = dir(sprintf('%s/yaleB%d/*.pgm', img_dir, i));    % Directories for individuals 10-38
        currdir = sprintf('%s/yaleB%d/', img_dir, i);
    end
    
    NumImages = length(dirc);  % Total images available in the directory
    
    % Load test images (remaining images after training)
    for j = N_train_per_indiv + 1:NumImages
        % Check if the index j is valid before accessing
        if j <= NumImages
            fname = sprintf('%s/%s', currdir, dirc(j).name);  % Construct the file name
            im = double(imread(fname));  % Read the test image

            iter = iter + 1;  % Increment image counter
            TestImages(:, iter) = im(:) - mean_A;  % Center the test image
            testid(iter) = i;  % Record the identity of the person
        end
    end
end

eigcoeffs_im = V' * TestImages;  % Compute eigencoefficients for test images

% Arrays to store the recognition rates
k_values = [1, 2, 3, 5, 10, 20, 30, 50, 55, 60, 65, 66, 67, 70, 75, 77, 100, 200, 300, 500, 1000];
rec_rate_values = zeros(length(k_values), 1);  % Recognition rates for all k values
rec_rate_remove3_values = zeros(length(k_values), 1);  % Recognition rates after removing 3 components

% Classification and accuracy calculation
for idx = 1:length(k_values)
    k = k_values(idx);  % Current value of k for recognition
    rec_rate = 0;  % Initialize recognition rate
    rec_rate_remove3 = 0;  % Initialize recognition rate (removing 3 components)
    
    Ntest = size(TestImages, 2);  % Dynamic test size
    
    for i = 1:Ntest        
        diffs = eigcoeffs_training - repmat(eigcoeffs_im(:, i), 1, N_train);  % Compute differences
        diffs1 = sum(diffs(1:k, :).^2, 1);  % Compute squared differences for first k eigenfaces
        [~, minindex] = min(diffs1);  % Find the nearest neighbor
        rec_rate = rec_rate + (testid(i) == person_id(minindex));  % Update recognition rate

        % Compute recognition rate while removing the first three eigenfaces
        diffs1 = sum(diffs(4:k+3, :).^2, 1);  % Compute squared differences excluding the first three
        [~, minindex] = min(diffs1);  % Find the nearest neighbor
        rec_rate_remove3 = rec_rate_remove3 + (testid(i) == person_id(minindex));  % Update rate
    end
    
    rec_rate = rec_rate / Ntest;  % Average recognition rate for k
    rec_rate_remove3 = rec_rate_remove3 / Ntest;  % Average recognition rate (after removing three)

    % Store the values for plotting
    rec_rate_values(idx) = 100 * rec_rate;  % Convert to percentage
    rec_rate_remove3_values(idx) = 100 * rec_rate_remove3;  % Convert to percentage
end

% Plot the recognition rates
figure;
plot(k_values, rec_rate_values, '-o', 'DisplayName', 'Rec rate');
hold on;
plot(k_values, rec_rate_remove3_values, '-x', 'DisplayName', 'Rec rate removing three');
xlabel('k');
ylabel('Recognition Rate (%)');
legend;  % Display legend
title('Recognition Rates for Different Values of k');  % Title for the plot
grid on;  % Add grid to the plot
