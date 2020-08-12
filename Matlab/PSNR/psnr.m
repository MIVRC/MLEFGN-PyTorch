clear;close all;
%% settings
folder_HR = 'HR';
folder_Denoise = 'MLEGN_Gray_15';

%% generate data
filepaths_HR = dir(fullfile(folder_HR,'*.png'));
filepaths_Denoise = dir(fullfile(folder_Denoise,'*.png'));

PSNR_all = zeros(1, length(filepaths_HR));
SSIM_all = zeros(1, length(filepaths_HR));

for idx_im = 1 : length(filepaths_HR)        
    im_GT = imread(fullfile(folder_HR,filepaths_HR(idx_im).name));    
    im_Denoise = imread(fullfile(folder_Denoise,filepaths_Denoise(idx_im).name));
    im_Denoise = rgb2gray(im_Denoise);
    [PSNR_all(idx_im), SSIM_all(idx_im)] = Cal_PSNRSSIM(im_GT, im_Denoise, 0, 0);
    
    fprintf('%d %s: PSNR= %f SSIM= %f\n', idx_im, filepaths_Denoise(idx_im).name, PSNR_all(idx_im), SSIM_all(idx_im));  
end

fprintf('--------Mean--------\n');
fprintf('PSNR= %f SSIM= %f\n', mean(PSNR_all), mean(SSIM_all));









