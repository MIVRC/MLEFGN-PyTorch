clear;close all;
%% settings
folder_HR = 'HR_RGB';

%% generate data
filepaths_HR = dir(fullfile(folder_HR,'*.png'));


for i = 1 : length(filepaths_HR)        
    im_hr = imread(fullfile(folder_HR,filepaths_HR(i).name));
    im_hr = im2double(im_hr);
    
    im_hr = rgb2gray(im_hr);
    
    
    HR_name = filepaths_HR(i).name;
    name = strcat(HR_name(1:end-4), '.png');
    filename = sprintf('DIV2K_train_HR/%s',name);
    imwrite(im_hr, filename);
end