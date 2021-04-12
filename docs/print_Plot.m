clear all, clc; 
% comb ={'case_3_bNDVI'}% {'case_11'}% {'case_7_Andrew'}%{'case_1', 'case_2', 'case_3_wo_Tau','case_3_bis','case_3',  'case_4','case_6_bis', 'case_5','case_7','case_8', 'case_9'}% {'VV', 'VH', 'VVaVH'} % 
% model_name = {'shallow_6_1580904831'}%{'Unet5','Linknet5','FPN5', 'shallow5','deeper5'}% {'LinkNet_6'} % ,'PSPNet'{'Unet3'}%{'Unet_despeck3_'}% {'Unet'}
inputs = {'Test_sigma_'}% , 'gamma_', 'beta_'};

comb = {'Tri'} % {'VH','VV','VVaVH' ,'Tri'} %    {'case_11'}% {'case__New_Andrew'}%{'case_1', 'case_2', 'case_3_wo_Tau','case_3_bis','case_3',  'case_4','case_6_bis', 'case_5','case__New','case_8', 'case_9'}% {'VV', 'VH', 'VVaVH'} % 
% model_name = {'FractalNet5', 'Unet5'}%{'Unet5','Linknet5','FPN5', 'shallow5','deeper5'}% {'LinkNet_6'} % ,'PSPNet'{'Unet3'}%{'Unet_despeck3_'}% {'Unet'}
model_name = {'FractalNet_New2'}%, 'shallow_CNN_New3', 'Unet_New3', 'Linknet_New3','SegNet_New2','FPN_New3'}%,'shallow_CNN_New'}%{ 'Unet6'} % {,'NestNet_New', 
%'Fractal_Net_New2', 
% k2 = [0,3,6,9]; [1, 4, 7, 10]; 
se = ones(3); 
for input_n = 1:length(inputs)
    input = inputs{input_n};
    for jj = 1:length(model_name) 
        for combin = 1:length(comb) 
         C = zeros(3); 
         for k = 0:9
%              k = k2(k1);
%% Aprire immagini per figure
%              figure(k+1); sgtitle(['fig. ',num2str(k + 1)]) 
%     %         figure(k+1 + 10*(combin-1));
%             input2 = imread(['C:\Users\massi\Downloads\segmentation_models-master\Tri_mobilenetv21_shallow_CNN_New2_VV_',num2str(k),'_wpatches128.tif']);
%             input1 = imread(['C:\Users\massi\Downloads\segmentation_models-master\Tri_mobilenetv21_shallow_CNN_New2_VH_',num2str(k),'_wpatches128.tif']);
%             inputN = imread(['C:\Users\massi\Downloads\segmentation_models-master\Tri_mobilenetv21_shallow_CNN_New2_NDVI_',num2str(k),'_wpatches128.tif']);
%             inputM = imread(['C:\Users\massi\Downloads\segmentation_models-master\Tri_mobilenetv21_shallow_CNN_New2_MNDWI_',num2str(k),'_wpatches128.tif']);
%             input3(:,:,1) = input2;
%             input3(:,:,2) = inputN;
%             input3(:,:,3) = inputM; % (input1 + eps)./(input2 + eps);
%             subplot(2,4,1); imshow(input3,[]); title('input')
           
%     %         subplot(2,2,1); imshow(input3,[]); title('input')
            target = imread(['C:\Users\massi\Downloads\segmentation_models-master\sigma_Tri_mobilenetv2_FPN_New2_target_',num2str(k),'_TH.tif']);
%             subplot(2,4,5); imshow(target,[]); title('target')
%     %         subplot(2,2,3); imshow(target,[]); title('target')

                output3 = imread(['C:\Users\massi\Downloads\segmentation_models-master\',input, comb{combin},'_mobilenetv2_',model_name{jj},'_output_',num2str(k),'_TH.tif']);
% %                 output = single(output3 > 0.5); 
%                 output = single(output3); 
%                 if jj < 4
%                     ind_plot = jj; %2*jj -1 ; % 
%                 else
%                     ind_plot = jj + 1;% 2*jj -1 ; %
%                 end
%                 subplot(2,4,ind_plot+1); imshow(output,[]); title(model_name{jj});
%     %             subplot(2,2,ind_plot+1); imshow(output,[]); title(model_name{jj});
%                 if k == 6
%                     imwrite(input3(64:128,1:64,:), ['C:\Users\massi\OneDrive\Desktop\Albufera_results\',input, comb{combin},'_mobilenetv2_',model_name{jj},'_input_',num2str(k),'_TH.png'])
%                     imwrite(output(64:128,1:64,:), ['C:\Users\massi\OneDrive\Desktop\Albufera_results\',input, comb{combin},'_mobilenetv2_',model_name{jj},'_output_',num2str(k),'_TH.png'])
%                     imwrite(target(64:128,1:64,:), ['C:\Users\massi\OneDrive\Desktop\Albufera_results\',input, comb{combin},'_target_',num2str(k),'_TH.png'])
%                 else
%                     imwrite(input3(:,1:64,:), ['C:\Users\massi\OneDrive\Desktop\Albufera_results\',input, comb{combin},'_mobilenetv2_',model_name{jj},'_input_',num2str(k),'_TH.png'])
%                     imwrite(output(:,1:64,:), ['C:\Users\massi\OneDrive\Desktop\Albufera_results\',input, comb{combin},'_mobilenetv2_',model_name{jj},'_output_',num2str(k),'_TH.png'])
%                     imwrite(target(:,1:64,:), ['C:\Users\massi\OneDrive\Desktop\Albufera_results\',input, comb{combin},'_target_',num2str(k),'_TH.png'])
%                 end
            se = ones(3);
            for band = 1:3
                for band2 = 1:3
                    y_o = output3(:,:,band); % imclose(output3(:,:,band), se); % 
   %     
                    y_t = target(:,:,band2);% imclose(target(:,:,band2), se);   %       .*targetf; %  
   
%                     y_o = imopen(y_o, se); %  output(:,:,k_band); %   
%                     y_t = imopen(y_t, se); 
                    tt = y_t.*y_o;

                    C(band,band2) = C(band, band2) + sum(tt(:));
                end
            end
         end
         C

     end
    end
end
