function globalHistogramEqualization(input_image)
    equalized_image = histeq(input_image); 
    imwrite(equalized_image, "GHE_LC1.png");
end