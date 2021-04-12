
for k = 0:9 
    
MNDWI = imshow(['D:\Albufera_2019_processed\Sigma2\',num2str(k),'MNDWI.tif']);
NDVI = imshow(['D:\Albufera_2019_processed\Sigma2\',num2str(k),'NDVI.tif']);
NDBI = imshow(['D:\Albufera_2019_processed\Sigma2\',num2str(k),'NDBI.tif']);

x(:,:,1) = 0.6*NDBI; 
x(:,:,2) = NDVI + 0.4*NDBI;
x(:,:,3) = MNDWI;

figure, 
imshow(x,[])
end