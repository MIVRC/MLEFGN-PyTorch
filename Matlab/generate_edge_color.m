clear;close all;
%% settings
folder_HR = 'DIV2K_train_HR';
folder_EDGE = 'DIV2K_train_EDGE';
scale = 1;

%% generate data
filepaths_HR = dir(fullfile(folder_HR,'*.png'));
filepaths_EDGE = dir(fullfile(folder_EDGE,'*.png'));


for i = 1 : length(filepaths_HR)        
    im_hr = imread(fullfile(folder_HR,filepaths_HR(i).name));
    im_hr = im2double(im_hr);
    
    im_hr_r = im_hr(:,:,1);
    im_hr_g = im_hr(:,:,2);
    im_hr_b = im_hr(:,:,3);


    [r_u1,r_u2] = gradient(im_hr_r);
    r_u1_m = r_u1./(sqrt(1 + r_u1.^2 + r_u2.^2));
    r_u2_m = r_u2./(sqrt(1 + r_u1.^2 + r_u2.^2));
    R_TD = divergence(r_u1_m, r_u2_m);
    
    
    [g_u1,g_u2] = gradient(im_hr_g);
    g_u1_m = g_u1./(sqrt(1 + g_u1.^2 + g_u2.^2));
    g_u2_m = g_u2./(sqrt(1 + g_u1.^2 + g_u2.^2));
    G_TD = divergence(g_u1_m, g_u2_m);
    
    [b_u1,b_u2] = gradient(im_hr_b);
    b_u1_m = b_u1./(sqrt(1 + b_u1.^2 + b_u2.^2));
    b_u2_m = b_u2./(sqrt(1 + b_u1.^2 + b_u2.^2));
    B_TD = divergence(b_u1_m, b_u2_m);


    RGB_TD = [];
    RGB_TD(:,:,1) = R_TD;
    RGB_TD(:,:,2) = G_TD;
    RGB_TD(:,:,3) = B_TD;

    filename = sprintf('DIV2K_train_EDGE/%s',filepaths_HR(i).name);
    imwrite(RGB_TD, filename);
end
