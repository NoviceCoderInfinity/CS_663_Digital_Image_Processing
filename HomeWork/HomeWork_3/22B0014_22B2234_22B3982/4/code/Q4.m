image = zeros(201, 201);
image(:, 101) = 255;

F = fft2(image);

F_shifted = fftshift(F);

log_magnitude = log(abs(F_shifted) + 1);

figure;
imagesc(log_magnitude);
colormap('jet');
colorbar;
title('Logarithm of the Fourier Magnitude');
