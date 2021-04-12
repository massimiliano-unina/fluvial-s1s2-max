clear all, close all, clc; 

% for k = 0:9 
%     
% target = imread(['C:\Users\massi\Downloads\segmentation_models-master\sigma_Tri_mobilenetv2_shallow_CNN7_target_',num2str(k),'_TH.tif'])
% output = imread(['C:\Users\massi\Downloads\segmentation_models-master\sigma_Tri_mobilenetv2_shallow_CNN7_output_',num2str(k),'_TH.tif'])
% 
% figure, subplot(121); imshow(target, [])
% subplot(122); imshow(output,[]) 
% sgtitle(['Data ' , num2str(k) ])
% end

load('C:\Users\massi\Downloads\segmentation_models-master\std_mean.mat')
load('C:\Users\massi\Downloads\segmentation_models-master\std_mean2.mat')
% load('C:\Users\massi\Downloads\segmentation_models-master\std_mean3.mat')

figure, plot(vvm_ts,'b--'), disp([' Wat - VV = ', num2str(mean(vvm_ts)), ' VH = ',num2str(mean(vhm_ts))])
hold on, plot(vhm_ts,'b-o')
hold on, plot(vvm_ts2,'g--'), disp([' No Veg/Wat - VV = ', num2str(mean(vvm_ts2)), ' VH = ',num2str(mean(vhm_ts2))])
hold on, plot(vhm_ts2,'g-o')

hold on, plot(vvn_ts,'r--'), disp([' Veg - VV = ', num2str(mean(vvn_ts)), ' VH = ',num2str(mean(vhn_ts))])
hold on, plot(vhn_ts,'r-o')
legend('vv wat', 'vh wat', 'vv no-wat/veg', 'vh no-wat/veg', 'vv veg','vh veg')
sgtitle('VV vs VH')

% figure, plot(vvm_ts3,'r--'), 
% hold on, plot(dvvm_ts3,'r-o')
% legend('VV', 'VH')
% hold on, plot(vvn_ts2,'b'), disp([' No Veg - VV = ', num2str(mean(vvn_ts2)), ' VH = ',num2str(mean(vhn_ts2))])
% hold on, plot(vhn_ts2,'bo')
% figure, plot(vvn_ts)