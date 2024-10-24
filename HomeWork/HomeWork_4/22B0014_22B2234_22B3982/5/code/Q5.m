clear; clc;

image_dir = '../../../Questionnare/ORL'; % Directory containing face images from ORL database

k_values = [1, 2, 5, 20, 50, 100]; % Different k values for which we will evaluate performance

N_train = 32*6; % Number of training images (32 individuals with 6 images each)
img_dim = 112*92; % Image dimensions (112x92 pixels)
data = zeros(img_dim, N_train); % Matrix to store the flattened image data for training
person_id = zeros(N_train, 1); % Vector to store the identity of each person (1 to 32)

iter = 0; % Initialize an iterator
for i = 1:32
    for j = 1:6
        img_path = fullfile(image_dir, ['s' num2str(i)], [num2str(j) '.pgm']); % Construct the image path
        img_data = double(imread(img_path)); % Read and convert image data to double precision
        iter = iter + 1; % Update iterator
        data(:, iter) = img_data(:); % Flatten and store the image data in column vector format
        person_id(iter) = i; % Record the identity of the person for the current image
    end
end

meanA = mean(data, 2); % Compute the mean of all training images
data = data - repmat(meanA, 1, N_train); % Subtract the mean from the training data to normalize

L = data' * data; % Compute L = A'A for eigendecomposition (more efficient than A*A')

[V, D] = eig(L); % Compute eigenvalues and eigenvectors of L
V = data * V; % Compute the eigenvectors of A*A' using the eigenvectors of L

% Normalize each eigenvector
for i = 1:N_train
    V(:, i) = V(:, i) / norm(V(:, i)); % Ensure each eigenvector is unit-normalized
end

% MATLAB returns eigenvectors in ascending order of eigenvalues, so we reverse the order
V = V(:, end:-1:1); % Reverse columns to get eigenvectors in descending order of eigenvalues

eigcoeffs_training = V' * data; % Compute eigencoefficients for training data

for idx = 1:length(k_values)
    k = k_values(idx); % Current k value for the evaluation

    % Initialize an array to store threshold values for each person
    taus = zeros(1, 32);

    % Calculate the verification threshold tau for each person based on intra-class distance
    for i = 1:32
        eigcoeffs_training_perperson = eigcoeffs_training(:, person_id == i); % Get eigencoefficients for person i
        dist = zeros(6, 6); % Initialize a distance matrix

        % Compute distances between all pairs of images for the same person
        for k1 = 1:6
            for k2 = 1:6
                dist(k1, k2) = sum(eigcoeffs_training_perperson(1:k, k1) - eigcoeffs_training_perperson(1:k, k2)).^2;
            end
        end
        taus(i) = max(dist(:)); % Set tau as the maximum intra-class distance for this person
    end

    tau = median(taus) * 0.5; % Final threshold is half the median of all individual taus
    fprintf('\nFor k = %d, threshold tau = %f', k, tau); % Display the threshold value for the current k

    % False Negative Rate (Images of known individuals that are incorrectly rejected)
    false_neg = 0; % Initialize the false negative count
    for i = 1:32
        for j = 7:10 % Test images (7th to 10th) for each known person
            img_path = fullfile(image_dir, ['s' num2str(i)], [num2str(j) '.pgm']); % Construct the image path
            img_data = double(imread(img_path)); % Read and convert image data to double precision
            eigcoeffs_im = V' * (img_data(:) - meanA); % Compute eigencoefficients for the test image

            % Calculate the distances between the test image and all training images
            diffs = eigcoeffs_training - repmat(eigcoeffs_im, 1, N_train);
            diffs = sum(diffs(1:k, :).^2, 1);
            [minval, minindex] = min(diffs); % Find the minimum distance

            if minval >= tau
                false_neg = false_neg + 1; % Increment false negative count if the distance is greater than tau
            end
        end
    end
    false_neg_rate = false_neg / (32 * 4); % Compute the false negative rate
    fprintf('\nFor k = %d, false negative rate = %f', k, 100 * false_neg_rate); % Display the false negative rate

    % False Positive Rate (Images of unknown individuals that are incorrectly accepted)
    false_pos = 0; % Initialize the false positive count
    for i = 33:40 % Test images of unknown individuals (33rd to 40th)
        for j = 1:10 % All 10 images per unknown individual
            img_path = fullfile(image_dir, ['s' num2str(i)], [num2str(j) '.pgm']); % Construct the image path
            img_data = double(imread(img_path)); % Read and convert image data to double precision
            eigcoeffs_im = V' * (img_data(:) - meanA); % Compute eigencoefficients for the test image

            % Calculate the distances between the test image and all training images
            diffs = eigcoeffs_training - repmat(eigcoeffs_im, 1, N_train);
            diffs = sum(diffs(1:k, :).^2, 1);
            [minval, minindex] = min(diffs); % Find the minimum distance

            if minval < tau
                false_pos = false_pos + 1; % Increment false positive count if the distance is less than tau
            end
        end
    end
    false_pos_rate = false_pos / (8 * 10); % Compute the false positive rate
    fprintf('\nFor k = %d, false positive rate = %f\n', k, 100 * false_pos_rate); % Display the false positive rate
end
