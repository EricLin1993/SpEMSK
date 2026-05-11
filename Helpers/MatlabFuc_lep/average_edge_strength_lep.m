function aes = average_edge_strength_lep(image)
    % Convert image to grayscale if it is not already
    
    image = squeeze(image);
    image = abs(NormalizedImage_lep(image));
    if size(image, 3) == 3
        image = rgb2gray(image);
    end
    
    % Detect edges using the Sobel operator
    edges = edge(image, 'sobel');

    % Compute the gradient magnitude
    [Gx, Gy] = imgradientxy(image, 'sobel');
    gradient_magnitude = sqrt(Gx.^2 + Gy.^2);

    % Calculate the average edge strength
    aes = mean(gradient_magnitude(edges));
end
