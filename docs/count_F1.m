close all, clear all, clc; 
% comb ={'case_3_bNDVI'}% {'case_11'}% {'case_7_Andrew'}%{'case_1', 'case_2', 'case_3_wo_Tau','case_3_bis','case_3',  'case_4','case_6_bis', 'case_5','case_7','case_8', 'case_9'}% {'VV', 'VH', 'VVaVH'} % 
% model_name = {'shallow_6_1580904831'}%{'Unet5','Linknet5','FPN5', 'shallow5','deeper5'}% {'LinkNet_6'} % ,'PSPNet'{'Unet3'}%{'Unet_despeck3_'}% {'Unet'}
inputs = {'Test_sigma_'}% , 'gamma_', 'beta_'};

comb = {'Tri'} % {'VH','VV','VVaVH','Tri'}% {'TriVH'}%  {'Tri'}% {'case_11'}% {'case__New_Andrew'}%{'case_1', 'case_2', 'case_3_wo_Tau','case_3_bis','case_3',  'case_4','case_6_bis', 'case_5','case__New','case_8', 'case_9'}% {'VV', 'VH', 'VVaVH'} % 
% model_name = {'FractalNet5', 'Unet5'}%{'Unet5','Linknet5','FPN5', 'shallow5','deeper5'}% {'LinkNet_6'} % ,'PSPNet'{'Unet3'}%{'Unet_despeck3_'}% {'Unet'}
model_name = {'Fractal_Net_New3'} % { 'Unet_New3'} %   { 'FractalNet_New2','DenseUNet_New3','Unet_New3', 'Linknet_New3','FPN_New3', 'SegNet_New2','shallow_CNN_New3'}%{ 'Unet6'} % {,'NestNet_New', 
%,'shallow_CNN_New2' 'Fractal_Net_New2', 
N_mont = 10% 17%  
F1_total = zeros(length(comb), length(model_name), N_mont);
Accuracy_total = zeros(length(comb), length(model_name),N_mont);
Precision_total = zeros(length(comb), length(model_name),N_mont);
Recall_total = zeros(length(comb), length(model_name),N_mont);
Time_total = zeros(length(comb), length(model_name));
for montec = 1:N_mont 
for input_n = 1:length(inputs)
    input = inputs{input_n};
 for thre = 10% 5:5:15%   10
thresh = thre*0.05;
for combin = 1:length(comb) 
    for jj = 1:length(model_name)   

        load([input, comb{combin},'_mobilenetv2_',model_name{jj},'_wvariables_',num2str(100*thresh),'_',num2str(montec),'.mat'])
        P = TP + FN
    end
end

% for combin = 1:length(comb) 
%     for jj = 1:length(model_name)   
%         disp([input,comb{combin},'_', model_name{jj}])
%         load(['C:\Users\massi\Downloads\segmentation_models-master\',input, comb{combin},'_mobilenetv2_',model_name{jj},'_times.mat'])
%         times1 = times/5
%     end
% end


for combin = 1:length(comb) 
    for jj = 1:length(model_name)   
%         load(['C:\Users\massi\Downloads\segmentation_models-master\',comb{combin},'_mobilenetv2_',model_name{jj},'_times.mat'])
        load([input, comb{combin},'_mobilenetv2_',model_name{jj},'_wvariables_',num2str(100*thresh),'_',num2str(montec),'.mat'])
        F1tot =  F1.*P;
        Accuracytot =  Accuracy.*P;
        Precisiontot =  Precision.*P;
        Recalltot =  Recall.*P;
        F1tot = sum(F1tot(:))/sum(P(:));%mean(F1(:)); %
        Accuracytot = sum(Accuracytot(:))/sum(P(:));
        Precisiontot = sum(Precisiontot(:))/sum(P(:));
        Recalltot = sum(Recalltot(:))/sum(P(:));
        F1_total(combin,jj, montec) = F1tot;
        Accuracy_total(combin,jj, montec) = Accuracytot;
%         ovAccuracy_total(combin,jj) = ovAccuracy;
        Precision_total(combin,jj, montec) = Precisiontot;
        Recall_total(combin,jj, montec) = Recalltot;
%         Time_total(combin,jj) = mean(times(2:10));
    end
end

file = fopen(['C:\Users\massi\Downloads\segmentation_models-master\',input,'Performance_',num2str(100*thresh),'.txt'], 'a');
fprintf(file, 'F1 \n');
fprintf(file,['Unet',' & Linknet',' & FPN','\n']); %,'Data2','&','Data3','&','Data4','&','Data5','&','Data6','&','Data_New'
dlmwrite(['C:\Users\massi\Downloads\segmentation_models-master\',input,'Performance_',num2str(100*thresh),'.txt'], F1_total,'delimiter','&','precision',4,'-append');
fprintf(file, 'Accuracy \n');
fprintf(file,['Unet',' & Linknet',' & FPN','\n']); %,'Data2','&','Data3','&','Data4','&','Data5','&','Data6','&','Data_New'
dlmwrite(['C:\Users\massi\Downloads\segmentation_models-master\',input,'Performance_',num2str(100*thresh),'.txt'], Accuracy_total,'delimiter','&','precision',4,'-append');
fprintf(file, 'Precision \n');
fprintf(file,['Unet',' & Linknet',' & FPN','\n']); %,'Data2','&','Data3','&','Data4','&','Data5','&','Data6','&','Data_New'
dlmwrite(['C:\Users\massi\Downloads\segmentation_models-master\',input,'Performance_',num2str(100*thresh),'.txt'], Precision_total,'delimiter','&','precision',4,'-append');
fprintf(file, 'Recall \n');
fprintf(file,['Unet',' & Linknet',' & FPN','\n']); %,'Data2','&','Data3','&','Data4','&','Data5','&','Data6','&','Data_New'
dlmwrite(['C:\Users\massi\Downloads\segmentation_models-master\',input,'Performance_',num2str(100*thresh),'.txt'], Recall_total,'delimiter','&','precision',4,'-append');
% fprintf(file, 'overall Accuracy \n');
% fprintf(file,['Unet',' & Linknet',' & FPN','\n']); %,'Data2','&','Data3','&','Data4','&','Data5','&','Data6','&','Data_New'
% dlmwrite(['C:\Users\massi\Downloads\segmentation_models-master\Performance.txt'], ovAccuracy_total,'delimiter','&','precision',4,'-append');
% fprintf(file, 'Time \n');
% fprintf(file,['Unet',' & Linknet',' & FPN','\n']); %,'Data2','&','Data3','&','Data4','&','Data5','&','Data6','&','Data_New'
% dlmwrite(['C:\Users\massi\Downloads\segmentation_models-master\Performance.txt'], Time_total,'delimiter','&','precision',4,'-append');

fprintf(file,'\n');
fclose(file);

 end
end
end 
F1_1 = sort(F1_total);
Precision_1 = sort(Precision_total);
Accuracy_1 = sort(Accuracy_total);
Recall_1 = sort(Recall_total);
disp( ['F1 mean' , num2str(mean(F1_1(7:end),3)), ' e std' , num2str(std(F1_1(7:end)))])
disp( ['Accuracy mean' , num2str(mean(Accuracy_1(7:end),3)), ' e std' , num2str(std(Accuracy_1(7:end)))]) 
disp( ['Precision mean' ,num2str(mean(Precision_1(7:end),3)) , ' e std' , num2str(std(Precision_1(7:end)))])
disp( ['Recall mean' ,num2str(mean(Recall_1(7:end),3)), ' e std' , num2str(std(Recall_1(7:end))) ])