%System akwizycji
clear all;
close all;
clc;
folder_path = 'C:\Users\Oliwier\Desktop\263629\Obrazy';
files = dir(fullfile(folder_path, '*.jpg'));

for file_idx = 1:length(files)
    folder_path = 'C:\Users\Oliwier\Desktop\263629\Obrazy';
    files = dir(fullfile(folder_path, '*.jpg'));
    oko = imread(fullfile(folder_path, files(file_idx).name));
    oko = imresize(oko, 1/2);
    oko = im2gray(oko);
    figure;
    imshow(oko)
    oko = double(oko);
    
    sx = [-1 0 1;
          -2 0 2;
          -1 0 1];
    sy = [-1 -2 -1;
          0 0 0;
          1 2 1];
    for i = 1:size(oko, 1) - 2 
        for j = 1:size(oko, 2) - 2 
            Gx = sum(sum(sx .* oko(i:i+2, j:j+2))); 
            Gy = sum(sum(sy .* oko(i:i+2, j:j+2))); 
            grad(i+1, j+1) = sqrt(Gx.^2 + Gy.^2);
        end
    end
    treshold = 30;
    grad = uint8(grad);
    grad = grad > treshold;
    
    [rows, cols] = size(grad);
    min_iris_radius = 31;
    max_iris_radius = 80;
    min_pupil_radius = 15;
    max_pupil_radius = 30;
    iris_accumulator = zeros(rows, cols, max_iris_radius);
    pupil_accumulator = zeros(rows, cols, max_pupil_radius);
    
    for y = 1:rows
        for x = 1:cols
            for r = min_iris_radius:max_iris_radius
                if grad(y, x) == 1
                    for theta = 0:pi/100:2*pi
                        a = round(x - r * cos(theta));
                        b = round(y - r * sin(theta));
                        if a > 0 && a <= cols && b > 0 && b <= rows
                            iris_accumulator(b, a, r) = iris_accumulator(b, a, r) + 1;
                        end
                    end
                end
            end
        end
    end
    
    for y = 1:rows
        for x = 1:cols
            if grad(y, x) == 1
                for r = min_pupil_radius:max_pupil_radius
                    for theta = 0:pi/100:2*pi
                        a = round(x - r * cos(theta));
                        b = round(y - r * sin(theta));
                        if a > 0 && a <= cols && b > 0 && b <= rows
                            pupil_accumulator(b, a, r) = pupil_accumulator(b, a, r) + 1;
                        end
                    end
                end
            end
        end
    end
    
    [max_iris_votes, max_iris_index] = max(iris_accumulator(:));
    [iris_y, iris_x, iris_r] = ind2sub(size(iris_accumulator), max_iris_index);
    [max_pupil_votes, max_pupil_index] = max(pupil_accumulator(:));
    [pupil_y, pupil_x, pupil_r] = ind2sub(size(pupil_accumulator), max_pupil_index);
    figure;
    imshow(grad);
    hold on;
    iris_theta = 0:0.01:(2*pi);
    pupil_theta = 0:0.01:(2*pi);
    plot(iris_x + iris_r * cos(iris_theta), iris_y + iris_r * sin(iris_theta), 'r', 'LineWidth', 2);
    plot(pupil_x + pupil_r * cos(pupil_theta), pupil_y + pupil_r * sin(pupil_theta), 'g', 'LineWidth', 2);
    title('Detekcja teczowki i zrenicy. Transformata Hougha');
    hold off;
    
    theta = -(0.38*pi):0.01:(1.25*pi);
    radii = linspace(pupil_r, iris_r, 128);
    [thetaGrid, radiiGrid] = meshgrid(theta, radii);
    x = pupil_x + radiiGrid .* cos(thetaGrid);
    y = pupil_y + radiiGrid .* sin(thetaGrid);
    normalizedIris = interp2(double(oko), x, y, 'linear', 0);
    figure;
    subplot(1, 2, 1);
    imshow(oko, []);
    title('Original Iris Image');
    subplot(1, 2, 2);
    imshow(normalizedIris, []);
    title('Normalized Iris Image');
    
    sigma = 5;
    lambda = 7;
    gamma = 1;
    orientation = 90;
    sizeoffilter = floor(3 * sigma);
    [gaborReal, gaborImag] = deal(zeros(2 * sizeoffilter + 1));
    for x = -sizeoffilter:sizeoffilter
        for y = -sizeoffilter:sizeoffilter
            xp = x * cos(orientation) + y * sin(orientation);
            yp = -x * sin(orientation) + y * cos(orientation);
            gaborReal(x + sizeoffilter + 1, y + sizeoffilter + 1) = exp(-(xp^2 + gamma^2 * yp^2) / (2 * sigma^2)) * cos(2 * pi * xp / lambda);
            gaborImag(x + sizeoffilter + 1, y + sizeoffilter + 1) = exp(-(xp^2 + gamma^2 * yp^2) / (2 * sigma^2)) * sin(2 * pi * xp / lambda);
        end
    end
    [height, width] = size(normalizedIris);
    segmentHeight = height / 16;
    segmentWidth = (width - 1) / 256;
    segments = zeros(segmentHeight, segmentWidth, 16 * 256);
    meanValues = zeros(1, 16 * 256);
    
    for i = 1:16
        for j = 1:256
            startRow = round((i - 1) * segmentHeight) + 1;
            endRow = round(i * segmentHeight);
            startCol = round((j - 1) * segmentWidth) + 1;
            endCol = round(j * segmentWidth);
            segment = normalizedIris(startRow:endRow, startCol:endCol);
            segments(:, :, ((i-1)*256)+j) = segment;
            meanValues(((i-1)*256)+j) = mean(segment(:));
        end
    end
    filteredReal = conv2(meanValues, gaborReal, 'same');
    filteredImag = conv2(meanValues, gaborImag, 'same');
    filteredImg = filteredReal + 1i * filteredImag;
    
    figure;
    imshow(real(reshape(filteredImg, [16, 256])), []);
    title('Filtered Image Real Part');
    figure;
    imshow(imag(reshape(filteredImg, [16, 256])), []);
    title('Filtered Image Imaginary Part');
    threshold = mean(filteredImg(:));
    binary_vec = filteredImg > threshold;
    
    binary_image = reshape(binary_vec, [16, 256]);
    figure;
    imshow(binary_image);
    title('binary image');
    iris_code_str = sprintf('%d', double(binary_vec));
    clear all;
    disp("Iteracja ukończona")
end

function [best_x0, best_y0, best_r] = daugman_circular(I, r_min, r_max)
    [rows, cols] = size(I);
    best_x0 = 0;
    best_y0 = 0;
    best_r = 0;
    max_val = -Inf;
    for x0 = 1:cols
        for y0 = 1:rows
            for r = r_min:r_max
                theta = 0:pi/180:2*pi;
                x = round(x0 + r * cos(theta));
                y = round(y0 + r * sin(theta));
                valid_idx = (x > 0) & (x <= cols) & (y > 0) & (y <= rows);
                x = x(valid_idx);
                y = y(valid_idx);
                sum_intensity = sum(I(sub2ind(size(I), y, x))) / length(x);
                if r > r_min
                    dI_dr = abs(sum_intensity - prev_sum_intensity);
                else
                    dI_dr = 0;
                end
                if dI_dr > max_val
                    max_val = dI_dr;
                    best_x0 = x0;
                    best_y0 = y0;
                    best_r = r;
                end
                prev_sum_intensity = sum_intensity;
            end
        end
    end
    figure, imshow(I, []);
    hold on;
    viscircles([best_x0, best_y0], best_r, 'EdgeColor', 'b');
    title(['Detected Circle: (', num2str(best_x0), ',', num2str(best_y0), ') with radius ', num2str

(best_r)]);
    hold off;
end
