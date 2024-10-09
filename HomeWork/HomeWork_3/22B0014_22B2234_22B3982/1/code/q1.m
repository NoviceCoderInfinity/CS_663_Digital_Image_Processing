img = imread('barbara256.png'); %load the image
img = double(img); % Convert to double for FFT processing
[M,N] = size(img); %here M=N=256

%pad the image to prevent wrap-around effect
%theoretically, the ideal low-pass filter has infinite dim in spatial domain (jinc)
%however, we can consider the size to be at least of size NxN and hence
%need to pad the original image to make it at least (2N-1 x 2N-1)
%for simplicity and symmetry, let's pad it to (2N x 2N)
pad_image =padarray(img, [N/2, N/2]);

%implement Ideal Low Pass Filter
%Take D (cut-off freq) to be 40, 60 and 80
%define a func to implement ILPF
function img_out = ILPF(img, D0) %D is cut-off freq
     fim2 = fftshift(fft2(img)); %compute FFT with zero-shift
     log_fim2 = log(abs(fim2)+1); % 1 added for stability
     figure; imshow(uint8(log_fim2), [min(log_fim2(:)) max(log_fim2(:))]); title("DTFT of Image"); %display Fourier Transform of image
     colormap("jet"); colorbar;
        
     [W1,W2] = size(fim2);                          %size of FFT image

     %define filter in freq-domain
     [u, v] = meshgrid(-W1/2:W1/2-1, -W2/2:W2/2-1); %u and v and 2D matrices
     D = sqrt(u.^2 + v.^2); 
     H = double(D <= D0);                           %LPF condition in 2D

     %display ILPF
     figure; imshow(log(1 + H), [min(log(1 + H(:))) max(log(1 + H(:)))]);
     colormap("jet"); colorbar; title('Fourier Transform of Ideal Low Pass Filter');
     
     %Multiply (point-wise) the two DTFTs
     filtered_img_F = fim2.*H;

     %display log-absolute plot of filtered image
     log_filtered = log(abs(filtered_img_F)+1);
     figure; imshow(uint8(log_filtered), [min(log_filtered(:)) max(log_filtered(:))]), colormap('jet'); colorbar; title("DTFT of filtered image")
     %IFFT
     img_out = ifft2(ifftshift(filtered_img_F));
end

%implement ILPF on image
%D=40
ilpf_40d = ILPF(pad_image,40);
ilpf_40d = ilpf_40d(N/2 +1: N/2 + N, N/2+1: N/2 +N); %selecting the central NxN image (w/o padding)
figure; 
imshow(uint8(ilpf_40d)); colormap("gray"); title("D=40 ILPF");


%Gaussian Low Pass Filter (GLPF)
%sigma (variance) is 40, 60 and 80
%define a func to implement GLPF

function img_out = GLPF(img, sigma)
    fim2 = fftshift(fft2(img)); %compute FFT with zero-shift
    [W1,W2] = size(fim2); %size of FFT image
    [u, v] = meshgrid(-W1/2:W1/2-1, -W2/2:W2/2-1);
    D = u.^2 + v.^2;
    H = exp(-D / (2 * sigma^2)); % Gaussian low-pass filter
    
    %display ILPF
    figure; imshow(log(1 + H), [min(log(1 + H(:))) max(log(1 + H(:)))]);
    colormap("jet"); colorbar; title('Fourier Transform of Gaussian Low Pass Filter');


    % Apply the filter in the frequency domain
    filtered_img_fft = fim2 .* H;

    %display log-absolute plot of filtered image
    log_filtered = log(abs(filtered_img_fft)+1);
    figure; imshow(uint8(log_filtered), [min(log_filtered(:)) max(log_filtered(:))]), colormap('jet'); colorbar; title("DTFT of filtered image")
    %IFFT
    img_out = ifft2(ifftshift(filtered_img_fft));
end

%Implement GLPF on image
%sigma=40
glpf_40sigma = GLPF(pad_image, 40);
glpf_40sigma = glpf_40sigma(N/2 +1: N/2 + N, N/2+1: N/2 +N); %selecting the central NxN image (w/o padding)

figure; 
imshow(uint8(glpf_40sigma)); colormap("gray"); title('sigma=40 GLPF');



