close all, clear all, clc; 
% comb ={'case_3_bNDVI'}% {'case_11'}% {'case_7_Andrew'}%{'case_1', 'case_2', 'case_3_wo_Tau','case_3_bis','case_3',  'case_4','case_6_bis', 'case_5','case_7','case_8', 'case_9'}% {'VV', 'VH', 'VVaVH'} % 
% model_name = {'shallow_6_1580904831'}%{'Unet5','Linknet5','FPN5', 'shallow5','deeper5'}% {'LinkNet_6'} % ,'PSPNet'{'Unet3'}%{'Unet_despeck3_'}% {'Unet'}
inputs = {'Test_sigma_'}% , 'gamma_', 'beta_'};

comb = {'Tri'}%{'VH','VV','VVaVH','Tri'}%   {'case_11'}% {'case__New_Andrew'}%{'case_1', 'case_2', 'case_3_wo_Tau','case_3_bis','case_3',  'case_4','case_6_bis', 'case_5','case__New','case_8', 'case_9'}% {'VV', 'VH', 'VVaVH'} % 
% model_name = {'FractalNet5', 'Unet5'}%{'Unet5','Linknet5','FPN5', 'shallow5','deeper5'}% {'LinkNet_6'} % ,'PSPNet'{'Unet3'}%{'Unet_despeck3_'}% {'Unet'}
model_name = {'Unet_New3'} %'Fractal_Net_New3'} % { 'FractalNet_New2','DenseUNet_New3','Unet_New3', 'Linknet_New3','FPN_New3', 'SegNet_New2','shallow_CNN_New3'}%{ 'Unet6'} % {,'NestNet_New', 
%,'shallow_CNN_New2' 'Fractal_Net_New2', 
for montec = 1:10%5 %17%%6%
se = ones(3); 
for input_n = 1:length(inputs)
    input = inputs{input_n};
for combin = 1:length(comb) 
for thre =10 % 5:5:15 % 
thresh = thre*0.05; 
TP = zeros(1,3); 
FP = TP; FN = TP; TN = TP; 

for jj = 1:length(model_name)   
    disp([input, '_', comb{combin}, '_', model_name{jj}])
    for k = 0:9%:891%0:206% 0:9:892 %0:24%259:2:289%196:5:271%1:5:11
        
        %% Opening Images
        output = imread(['C:\Users\massi\Downloads\segmentation_models-master\images\results_Metri2\',input, comb{combin},'_mobilenetv2',num2str(montec),'_',model_name{jj},'_output_',num2str(k),'_wpatches128.tif']); % _wpatches128
%         output = imread(['C:\Users\massi\Downloads\segmentation_models-master\',input, comb{combin},'_mobilenetv21_',model_name{1},'_output_',num2str(k),'_wpatches128.tif']);
%         randout = randn(size(output));
%         maskrand = imdilate(single(randout > 0 ), ones(5));
%         if jj ==2
%             output = output + 0.2*randout.*(maskrand );
%             output = imgaussfilt(output,2);
%             output = output.*(output > 0.5); 
%         elseif jj == 3
%             output = output + 0.3*randout.*(maskrand );
%             output = imgaussfilt(output,2);
%             output = output.*(output > 0.5); 
%         elseif jj == 4
%             output = output + 0.4*randout.*(maskrand );
%             output = imgaussfilt(output,2);
%             output = output.*(output > 0.5); 
%         elseif jj == 5
%             output = output + 0.6*randout.*(maskrand );
%             output = imgaussfilt(output,2);
%             output = output.*(output > 0.5); 
%         elseif jj == 6
%             output = output + 0.5*randout.*(maskrand );
%             output = imgaussfilt(output,2);
%             output = output.*(output > 0.5); 
% 
%         end
        target = imread(['C:\Users\massi\Downloads\segmentation_models-master\images\results_Metri2\',input, comb{combin},'_mobilenetv21_',model_name{jj},'_target_',num2str(k),'_wpatches128.tif']);% _wpatches128
        output3 = output; 
save_tif(target,['C:\Users\massi\Downloads\segmentation_models-master\images\results_Metri2\',input, comb{combin},'_mobilenetv21_',model_name{jj},'_output_',num2str(k),'_wpatches128.tif'],['C:\Users\massi\Downloads\segmentation_models-master\',input, comb{combin},'_mobilenetv2_',model_name{jj},'_target_',num2str(k),'_TH.tif'])

save_tif(output3,['C:\Users\massi\Downloads\segmentation_models-master\images\results_Metri2\',input, comb{combin},'_mobilenetv21_',model_name{jj},'_output_',num2str(k),'_wpatches128.tif'],['C:\Users\massi\Downloads\segmentation_models-master\',input, comb{combin},'_mobilenetv2_',model_name{jj},'_output_',num2str(k),'_TH.tif'])
        
        
        %% Thresholding Output
% output3 = single(output > thresh); 
%% Saving images
%% Compute False/True Negative/Positive
for k_band = 1:3
  y_o = imclose(output3(:,:,k_band), se); % output3(:,:,k_band); % 
   %     
   y_t = imclose(target(:,:,k_band), se);   % target(:,:,k_band);%      .*targetf; %  
   
   y_o = imopen(y_o, se); %  output(:,:,k_band); %   
   y_t = imopen(y_t, se);   %  target(:,:,k_band);%.*targetf; % 
   o3(:,:,k_band) = y_o;
   t3(:,:,k_band) = y_t;
TruePositive_1 = y_o.*y_t;%.*targetf;%).*targetf;%
FalsePositive_1 = (y_o.*(1 - y_t));%.*targetf;%).*targetf;%
TrueNegative_1 = ((1 - y_o).*(1 - y_t));%.*targetf;%).*targetf;%
FalseNegative_1 = ((1 - y_o).*(y_t));%.*targetf;%).*targetf;%
TP(k_band) = TP(k_band) +  sum(TruePositive_1(:));
FP(k_band) = FP(k_band) + sum(FalsePositive_1(:));
FN(k_band) = FN(k_band) + sum(FalseNegative_1(:));
TN(k_band) = TN(k_band) + sum(TrueNegative_1(:));
end

%% Compute Main Considered Metrics (Accuracy, Precision, Confusion Matrix, F1-score) 

for i = 1:3
    Accuracy(i) = (TP(i) + TN(i) + eps)/(TP(i) + TN(i) + FP(i) + FN(i) + eps);
    Precision(i) = (TP(i) + eps)/(TP(i) + FP(i) + eps);
    Recall(i) = (TP(i)+eps)/(TP(i) + FN(i) + eps);
    Conf(:,:,i) = [TP(i), FP(i); FN(i), TN(i)];
    F1(i) = (2*Precision(i)*Recall(i) + eps)/(Precision(i)+Recall(i) + eps);
    T = target(:,:,i); 
    Elem(i) = sum(T(:));
end
end
save([input, comb{combin},'_mobilenetv2_',model_name{jj},'_wvariables_',num2str(100*thresh),'_',num2str(montec),'.mat'],'Elem', 'F1', 'Accuracy', 'Precision', 'Recall','Conf','TN','FN','TP','FP');
end
end
end
end
end