clear all, close all, clc; 

park = imread("D:\Albufera_2019_processed\subset_albufera\Dataset\park2.tif");
[M,N] = size(park); 
wat2 = zeros(M,N); 

for k = 1:12
    
    wat = imread(['D:\Albufera_2019_processed\subset_albufera\Dataset\' , num2str(k) , 'Wat_Thre.tif']);
    wat2 = wat2 + wat; 

end

disp(num2str(max(wat2(:))))
wat3 = wat2 > 6; 
se = ones(5);
P = 1000; 
wat3 = bwareaopen(wat3,P);
% wat3 = imopen(wat3, se); 
% wat3 = imclose(wat3, se); 
save_tif(wat3, "D:\Albufera_2019_processed\subset_albufera\Dataset\park2.tif", "D:\Albufera_2019_processed\subset_albufera\Dataset\Lake.tif")
figure, 
imshow(wat3, []) 