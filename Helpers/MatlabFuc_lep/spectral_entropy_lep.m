function se = spectral_entropy_lep(image)

%  Enping Lin, 2024.6.11
% ========================================================


    image = abs(NormalizedImage_lep(image));

    % Convert image to grayscale if it is not already
    if size(image, 3) == 3
        image = rgb2gray(image);
    end
    

    % Compute the Fourier Transform of the image
    F = fft2(image);
    F = abs(F).^2; % Power spectrum

    % Normalize the power spectrum
    F = F / sum(F(:));

    % Compute the entropy
    se = -sum(F(:) .* log2(F(:) + eps)); % Adding eps to avoid log(0)

end
