clear;close all;
%% settings
folder_HR = 'DIV2K_train_HR';
folder_EDGE = 'DIV2K_train_LR_bicubic/50';
noise_std = 50;

%% generate data
filepaths_HR = dir(fullfile(folder_HR,'*.png'));


for i = 1 : length(filepaths_HR)        
    im_hr = imread(fullfile(folder_HR,filepaths_HR(i).name));
    im_hr = im2double(im_hr);
    
    
    noise_image = imnoise(im_hr,'gaussian',0,(noise_std/255)^2);
    
    HR_name = filepaths_HR(i).name;
    name = strcat(HR_name(1:end-4), 'x1.png');
    filename = sprintf('DIV2K_train_LR_bicubic/50/%s',name);
    imwrite(noise_image, filename);
end