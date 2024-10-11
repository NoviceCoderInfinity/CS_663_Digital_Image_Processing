
tic;
img1 = imread('barbara256.png');
img2 = imread('kodak24.png');

img1 = im2double(img1);
img2 = im2double(img2);

%adding gaussian noise to images
sigma_n = 5; % std dev of the noise added
noise_1 = (sigma_n/255)*randn(size(img1)); %property of variance : Var(aX) = a^2Var(X), randn() creates noise of mean =0, std-dev=1
noise_2 = (sigma_n/255)*randn(size(img2));

img1_noisy = img1 + noise_1;
img2_noisy = img2 + noise_2;

% mean-shift filtering

sigma_s = 15; sigma_r = 3;
img1_mean_shift = mean_shift_filter(img1_noisy,sigma_s,sigma_r/255); %since im2double has normalized intensity b/w 0 and 1, we divide sigma by 255
img2_mean_shift = mean_shift_filter(img2_noisy,sigma_s,sigma_r/255);

figure(1);
imshow(img1_noisy); title("barbara image with noise");
imshow(img1_mean_shift); title("barbara image mean-shift filtered"); 
imshow(img2_noisy); title("kodak image with noise");
imshow(img2_mean_shift); title("kodak image mean-shift filtered");

function filtered_image = mean_shift_filter(I,sigma_s,sigma_r)
    [r, c] = size(I); % rows and columns in Image I

    BW_half = ceil(3*sigma_s)+1; % BW = 6*sigma_s + 1, neighbourhood window span (as stated in slides: mean-shift 
    % for clustering : speed-up,
    % hence we take a predefined window for computing the summations

    e = 0.01; % accuracy threshold

    filtered_image = zeros(size(I)); %intialize the filtered image with zero matrix

    for x = 1:c %loop through each row
        for y = 1:r %loop through each column
            f = [x y I(y,x)]; % feature vector, since we have only one channel and not RGB
            while 1
                i1 = max(y-BW_half,1); i2 = min(y+BW_half,r); % neighbourhood start and end rows
                j1 = max(x-BW_half,1); j2 = min(x+BW_half,c); % neighbourhood start and end columns
                LI = I(i1:i2,j1:j2); %image chunck on which summation is performed
                [X, Y] = meshgrid(j1:j2,i1:i2);
                Gs = exp(-1*((X-f(1)).^2+(Y-f(2)).^2)/(2*sigma_s^2)); % spatial gaussian weights
                Gr = exp(-1*((LI-f(3)).^2)/(2*sigma_r^2)); % range gaussian weights
                G = Gs.*Gr;
                Wp = sum(G,'all');
                fx = sum(G.*X,'all')/Wp;
                fy = sum(G.*Y,'all')/Wp;
                fI = sum(G.*LI,'all')/Wp;
                if norm(f-[fx fy fI])> e
                    f = [fx fy fI];
                else 
                    break;
                end
            end
            filtered_image(y,x) = f(3);
        end
    end     
end

toc;
elapsed_time = toc;
