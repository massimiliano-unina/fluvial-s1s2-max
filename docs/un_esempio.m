clear all, close all, clc; 

methods = {'case_9', 'case_8', 'case_5', 'case_6_bis'}
methods1 = {'case 9 ( without \rho_6) ', 'case 8 ( without \rho_{LT} and \tau', 'case 5 ( all inputs) ', 'case 6 (without \tau)'}

patch_n = { '20', '100', '200', '150', '55'}
for kkk = 1:length(patch_n)
    figure, 
    input = imread(['C:\Users\massi\Downloads\segmentation_models-master\images\results\case_5_mobilenetv2_shallow_6_rhoLT_', patch_n{kkk}, '_wpatches128.tif'],1);
    target = imread(['C:\Users\massi\Downloads\segmentation_models-master\images\results\case_5_mobilenetv2_shallow_6_target_', patch_n{kkk}, '_wpatches128.tif'],1);
    subplot(2,3,1); imshow(input,[]) ; title('input'); 
    blu = ( 1 - target(:,:,1) - target(:,:,2) - target(:,:,3)) ;
    B1(:,:,1) = target(:,:,1) + 0.8*target(:,:,3); 
    B1(:,:,2) = target(:,:,2) + 0.2*target(:,:,3);
    B1(:,:,3) =  blu ;

    subplot(2,3,2); imshow(B1,[]) ; title('target'); 
    for kk = 1:length(methods)

        aaa3 = imread(['C:\Users\massi\Downloads\segmentation_models-master\images\results\', methods{kk}, '_mobilenetv2_shallow_6_output_', patch_n{kkk}, '_wpatches128.tif'],1);
        aaa2 = max(aaa3, [], 3); 
        aaa(:,:,1) = aaa3(:,:,1) == aaa2; 
        aaa(:,:,2) = aaa3(:,:,2) == aaa2; 
        aaa(:,:,3) = aaa3(:,:,3) == aaa2; 
        
        blu2 = ( 1 - aaa(:,:,1) - aaa(:,:,2) - aaa(:,:,3));

        B(:,:,1) = aaa(:,:,1) + 0.8*aaa(:,:,3); 
        B(:,:,2) = aaa(:,:,2) + 0.2*aaa(:,:,3);
        B(:,:,3) =  blu2;
        subplot(2,3,kk+2); imshow(B,[]) ; title(methods1{kk}); 
    end
%     sgtitle(patch_n{kkk}) 
end

figure, 
a(:,:,1) = zeros(12);
a(:,:,2) = zeros(12);
a(:,:,2) = zeros(12);
b = ones(3,12); 
a(1:3,:,2) = 0.2*b ; 
a(1:3,:,1) =  0.8*b; 
a(4:6, :,1) = b;
a(7:9, : ,3) = b;
a(10:12, : , 2) = b; 
figure, imshow(a,[])