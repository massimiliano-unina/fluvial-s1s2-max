clear all, close all, clc; 
 park = imread("D:\Albufera_2019_processed\Dataset_paper_2020\park2.tif");
 [M,N] = size(park); 
 
 sar2 = zeros(M,N); 
  opt2 = zeros(M,N); 

for k = 1:10
   opt = imread(['D:\Albufera_2019_processed\Dataset_paper_2020\',num2str(k-1),'MNDWI.tif']);
    sar = imread(['D:\Albufera_2019_processed\Dataset_paper_2020\',num2str(k),'VV_Thre.tif']);
    figure, hist(sar(:), 2^8);
    opt2 = opt2 + opt;
    sar2 = sar2 + sar; 
end
figure,subplot(121), imshow(sar2.*park, []); 
subplot(122), imshow(opt2.*park, []); 