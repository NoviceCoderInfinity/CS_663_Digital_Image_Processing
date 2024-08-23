%Functions
function ncc_value = computeNCC(J1, J4)
    [w, h] = size(J1);
    m1 = sum(J1(:)) / (w * h);
    m4 = sum(J4(:)) / (w * h);
    n = 0;
    d1 = 0;
    d4 = 0;
    
    for r = 1:w
        for c = 1:h
            diff1 = J1(r, c) - m1;
            diff4 = J4(r, c) - m4;
            n = n + (diff1 * diff4);
            d1 = d1 + (diff1 ^ 2);
            d4 = d4 + (diff4 ^ 2);
        end
    end
    
    d = sqrt(d1 * d4);
    
    if d == 0
        ncc_value = 0;
    else
        ncc_value = n / d;
    end
end

function [JE, QMI] = computeJointHist(J1, J4, binWidth)
    minValue = 0;
    maxValue = 255;
    numBins = ceil((maxValue - minValue + 1) / binWidth);
    jointHist = zeros(numBins, numBins);
    [rows, cols] = size(J1);
    
    for r = 1:rows
        for c = 1:cols
            binJ1 = floor(J1(r, c) / binWidth) + 1;
            binJ4 = floor(J4(r, c) / binWidth) + 1;
            jointHist(binJ1, binJ4) = jointHist(binJ1, binJ4) + 1;
        end
    end

    jointHistNormalized = jointHist / sum(jointHist(:));
    Hist1 = sum(jointHistNormalized, 2);
    Hist4 = sum(jointHistNormalized, 1);
    JE = 0;
    QMI = 0;

    for i1 = 1:numBins
        for i2 = 1:numBins
            pij = jointHistNormalized(i1, i2);
            pipj = Hist1(i1) * Hist4(i2);
            if pij > 0
                JE = JE + pij * log2(pij);
            end
            QMI = QMI + (pij - pipj)^2;
        end
    end

    JE = -JE;
end

function Joint = JointHist(J1, J4, binWidth)
    minValue = 0;
    maxValue = 255;
    numBins = ceil((maxValue - minValue + 1) / binWidth);
    jointHist = zeros(numBins, numBins);
    [rows, cols] = size(J1);
    
    for r = 1:rows
        for c = 1:cols
            binJ1 = floor(J1(r, c) / binWidth) + 1;
            binJ4 = floor(J4(r, c) / binWidth) + 1;
            jointHist(binJ1, binJ4) = jointHist(binJ1, binJ4) + 1;
        end
    end

    Joint = jointHist / sum(jointHist(:));
end
% Main Script
J1 = imread('T1.jpg');
J1 = double(J1);

J2 = imread('T2.jpg');
J2 = double(J2);

J3 = imrotate(J2, 28.5, 'bilinear', 'crop');

%figure, imshow(J1/255);
%title('J1');

%figure, imshow(J2/255);
%title('J2');

%figure, imshow(J3/255);
%title('J3');

angles = -45:1:45;
nAngles = length(angles);
NCCs = zeros(nAngles, 1);
JEs = zeros(nAngles, 1);
QMIs = zeros(nAngles, 1);


for i = 1:nAngles
    angle = angles(i);
    J4 = imrotate(J3, angle, 'bilinear', 'crop');
    NCCs(i) = computeNCC(J1, J4);
    [JEs(i), QMIs(i)] = computeJointHist(J1, J4, 10);
end

% Plot NCC versus theta
figure;
plot(angles, NCCs, '-o', 'LineWidth', 2);
xlabel('Theta');
ylabel('NCC');
title('NCC vs Theta');
grid on;
saveas(gcf, 'NCC_vs_Theta.png');

% Plot JE versus theta
figure;
plot(angles, JEs, '-o', 'LineWidth', 2);
xlabel('Theta');
ylabel('JE');
title('JE vs Theta');
grid on;
saveas(gcf, 'JE_vs_Theta.png');

% Plot QMI versus theta
figure;
plot(angles, QMIs, '-o', 'LineWidth', 2);
xlabel('Theta');
ylabel('QMI');
title('QMI vs Theta');
grid on;
saveas(gcf, 'QMI_vs_Theta.png');

optimalNCCAngle = angles(NCCs == max(NCCs));
optimalJEAngle = angles(JEs == min(JEs));
optimalQMIAngle = angles(QMIs == max(QMIs));

fprintf('Optimal rotation angle based on NCC: %.1f degrees\n', optimalNCCAngle);
fprintf('Optimal rotation angle based on JE: %.1f degrees\n', optimalJEAngle);
fprintf('Optimal rotation angle based on QMI: %.1f degrees\n', optimalQMIAngle);

J4 = imrotate(J3, -29, 'bilinear', 'crop');
figure;
Joint = JointHist(J1, J4, 10);
imagesc(Joint);
axis xy;
colorbar;

title('Joint Histogram');
xlabel('J1 Intensity');
ylabel('J4 Intensity');