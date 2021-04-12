close all, clear all, clc; 
k = 5; 

VV = single(imread("C:\Users\massi\Downloads\Sigma0_VV.tif"));
VH = single(imread("C:\Users\massi\Downloads\Sigma0_VH.tif"));
RG1(:,:,1) = VH; 
RG1(:,:,2) = VV;
VV_VH = (VV + eps)./(VH + eps);
% RG1(:,:,3) = VV_VH;
% Agray = rgb2gray(RG1);
imageSize = size(RG1)

%% Entropy Filter
RG = single(entropyfilt(RG1));
RG_1 = RG(:,:,1); 
RG_2 = RG(:,:,2); 
% RG_3 = RG(:,:,3); 

X(:,1) = VV(:);
X(:,2) = VH(:); 
X(:,3) = VV_VH(:);
X(:,4) = RG_1(:);
X(:,5) = RG_2(:); 
% X(:,6) = RG_3(:);
% X(:,7) = f_2D_1(:);
% X(:,8) = f_2D_2(:);
% X(:,9) = f_2D_3(:);

X = bsxfun(@minus, X, mean(X));
X = bsxfun(@rdivide,X,std(X));

coeff = pca(X);
% Out(:,:,1) = reshape(X*coeff(:,1),imageSize(1),imageSize(2));
% Out(:,:,2) = reshape(X*coeff(:,2),imageSize(1),imageSize(2));
% Out(:,:,3) = reshape(X*coeff(:,3),imageSize(1),imageSize(2));
Out(:,:,1) = VV;
Out(:,:,2) = VH; 
% Out(:,:,3) = VV_VH;
Out(:,:,4) = RG_1;
Out(:,:,3) = RG_2; 
% Out(:,:,3) = RG_3;
%% K-means and analysis
L = imsegkmeans(Out,k,'NormalizeInput',true,'NumAttempts', 10);
% L = watershed(Out);
RG1(:,:,3) = VV_VH;

B = labeloverlay(RG1,L);
A = unique(L);
figure, 
for i = 1:length(A)
    th = A(i); 
    D = L == th; 
    subplot(1,k,i); imshow(D, []) ;
    save_tif(single(D),"C:\Users\massi\Downloads\Sigma0_VH.tif","C:\Users\massi\Downloads\feature_" + num2str(i) + ".tif")
end
figure, 
imshow(B)
title('Labeled Image')
% 



