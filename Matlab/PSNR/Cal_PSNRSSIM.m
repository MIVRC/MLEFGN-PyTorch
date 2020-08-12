function [psnr_cur, ssim_cur] = Cal_PSNRSSIM(A,B,row,col)


[n,m,ch]=size(B);
A = A(row+1:n-row,col+1:m-col,:);
B = B(row+1:n-row,col+1:m-col,:);
A=double(A); % Ground-truth
B=double(B); %

e=A(:)-B(:);
mse=mean(e.^2);
psnr_cur=10*log10(255^2/mse);

if ch==1
    [ssim_cur, ~] = ssim_index(A, B);
else
    ssim_cur = (ssim_index(A(:,:,1), B(:,:,1)) + ssim_index(A(:,:,2), B(:,:,2)) + ssim_index(A(:,:,3), B(:,:,3)))/3;
end

end