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

    [r_u1,r_u2] = gradient(im_hr);
    r_u1_m = r_u1./(sqrt(1 + r_u1.^2 + r_u2.^2));
    r_u2_m = r_u2./(sqrt(1 + r_u1.^2 + r_u2.^2));
    EDGE = divergence(r_u1_m, r_u2_m);

    filename = sprintf('DIV2K_train_EDGE/%s',filepaths_HR(i).name);
    imwrite(EDGE, filename);
end
