% clear all, close all, clc; 
clear all; close all; clc; 

comb = {'VVaVH'}%'VH', 'VV','VVaVH'}%, 'VVaVHaSum','VVaVHaRatio','VVaVHaDiff','Total'}%{'VVaVH'}% 
model_name ={'Unet4','Linknet4','FPN4'}% {'Unet4'}%,'PSPNet'{'Unet3'}%{'Unet_despeck3_'}% {'Unet'}

for combin = 1:length(comb) 
    for jj = 1:length(model_name)   

        load([comb{combin},'_mobilenetv2_',model_name{jj},'_wvariables_50.mat'])

details = [1,5,6,8];%,8];
for kk = 1:length(details)% k = 9:25:25%0:24%259:2:289%196:10:271%1:5:11%1:6:49%2:40%8
    k = details(kk);
inp = imread(['C:\Users\massi\Downloads\segmentation_models-master\',comb{combin},'_mobilenetv2_',model_name{jj},'_VV_',num2str(k-1),'_wpatches128.tif']);
inp2 = imread(['C:\Users\massi\Downloads\segmentation_models-master\',comb{combin},'_mobilenetv2_',model_name{jj},'_VH_',num2str(k-1),'_wpatches128.tif']);

% tar = imread(['C:\Users\massi\Downloads\segmentation_models-master\',comb{combin},'_mobilenetv2_',model_name{jj},'_target_',num2str(k-1),'_wpatches128.tif']);
% out2 = imread(['C:\Users\massi\Downloads\segmentation_models-master\',comb{combin},'_mobilenetv2_',model_name{jj},'_output_',num2str(k-1),'_wpatches128.tif']);
% out2 = single(out > 0.5); 
% %tarb = imread(['C:\Users\massi\Downloads\segmentation_models-master\VVaVH_mobilenetv2_Unet3_target_',num2str(k-1),'_wpatches128.tif']);
% %outb = imread(['C:\Users\massi\Downloads\segmentation_models-master\VVaVH_mobilenetv2_Unet3_output_',num2str(k-1),'_wpatches128.tif']);
% %outb2 = single(outb > 0.5); 
if kk == 1
% tar = tar(68:115,33:84,:);
inp1(:,:,1) = inp2(68:115,33:84);
inp1(:,:,2) = inp(68:115,33:84);
inp1(:,:,3) = inp2(68:115,33:84)./inp(68:115,33:84);
% out2 = out2(68:115,33:84,:);
elseif kk==2
% tar = tar(93:124,51:87,:);
inp1(:,:,1) = inp2(93:124,51:87);
inp1(:,:,2) = inp(93:124,51:87);
inp1(:,:,3) = inp2(93:124,51:87)./inp(93:124,51:87);
% inp = inp(93:124,51:87);
% inp2 = inp2(93:124,51:87);
% out2 = out2(93:124,51:87,:);
elseif kk==3
%     tar = tar(57:77,26:76,:);

inp1(:,:,1) = inp2(57:77,26:76);
inp1(:,:,2) = inp(57:77,26:76);
inp1(:,:,3) = inp2(57:77,26:76)./inp(57:77,26:76);
% inp = inp(57:77,26:76);
% out2 = out2(57:77,26:76,:);
elseif kk==4
%     tar = tar(11:33,64:115,:);
inp1(:,:,1) = inp2(11:33,64:115);
inp1(:,:,2) = inp(11:33,64:115);
inp1(:,:,3) = inp2(11:33,64:115)./inp(11:33,64:115);
% inp = inp(11:33,64:115);
% out2 = out2(11:33,64:115,:);
elseif kk==5
%     tar = tar(68:115,33:84,:);
inp1(:,:,1) = inp2(68:115,33:84);
inp1(:,:,2) = inp(68:115,33:84);
inp1(:,:,3) = inp2(68:115,33:84)./inp(68:115,33:84);
inp = inp(68:115,33:84);
% out2 = out2(68:115,33:84,:);
end
% figure,
% subplot(221); imshow(inp,[]); title('input')
% % subplot(242); imshow(out,[])
% subplot(222); imshow(out2,[]); title('output Despeckling')
% subplot(223); imshow(tar,[]); title('target')
% 
% subplot(224); imshow(out2.*tar,[]); title('Black for errors')
% % subplot(245); imshow(tarb,[])
% a = axes;
% t1 = title([comb{combin},' Mobilenetv2 ',model_name{jj},' Output ',num2str(k-1)]);
% a.Visible = 'off'; % set(a,'Visible','off');
% t1.Visible = 'on'; % set(t1,'Visible','on');
%subplot(234); imshow(inp,[]); title('input')
%subplot(235); imshow(outb2,[]); title('output No-%Despeckling')
% subplot(236); imshow(outb2.*tarb,[]); title('Black for errors')
inp = (inp1 - min(inp1(:)))./(max(inp1(:)) - min(inp1(:)));
imwrite(inp,['C:\Users\massi\Downloads\segmentation_models-master\',comb{combin},'_mobilenetv2_',model_name{jj},'_RGB_',num2str(k-1),'_wpatches128.png'],'PNG','BitDepth',8)
% imwrite(out2,['C:\Users\massi\Downloads\segmentation_models-master\',comb{combin},'_mobilenetv2_',model_name{jj},'_output_',num2str(k-1),'_wpatches128.png'],'PNG','BitDepth',8)
% imwrite(tar,['C:\Users\massi\Downloads\segmentation_models-master\',comb{combin},'_mobilenetv2_',model_name{jj},'_target_',num2str(k-1),'_wpatches128.png'],'PNG','BitDepth',8)
end
end
end