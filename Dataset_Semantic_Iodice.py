# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:22:05 2020

@author: massi
"""
import os 
import numpy as np 
import imageio 
from matplotlib import pyplot as plt 
from tifffile import imsave
path = r"D:\Albufera_2019_processed\subset_albufera\s2\\"
path2 = r"C:\Users\massi\OneDrive\Desktop\snap_albufera_beta_gamma_sigma\\"  #"D:\Albufera_2019_processed\subset_albufera\s1\\"
path3 = r"D:\Albufera_2019_processed\subset_albufera\Dataset\\"

if not os.path.exists(path3):
    os.makedirs(path3)
    
dir_list = os.listdir(path2)
dir_list.sort()

#file_inr = os.path.join(path, "Subset_S2B_MSIL2A_20191207T105329_N0213_R051_T31SBD_20191207T114211_resampled.data\B8.img")
file_inr = r"D:\Albufera_2019_processed\subset_albufera\Dataset\20190506_NDVI_pre.tif"

cc = 0
for folder_1 in dir_list: 
   if folder_1.find(".dim") == -1: 
#        print(folder_1)
#        if os.path.exists(os.path.join(path2, folder_1, "Sigma0_VV.img")):  
#            cc += 1
#print(cc)
       file_VH = os.path.join(path2, folder_1, "Sigma0_VH.img")
       file_VV = os.path.join(path2, folder_1, "Sigma0_VV.img")
       file_bVH = os.path.join(path2, folder_1, "Beta0_VH.img")
       file_bVV = os.path.join(path2, folder_1, "Beta0_VV.img")
       file_gVH = os.path.join(path2, folder_1, "Gamma0_VH.img")
       file_gVV = os.path.join(path2, folder_1, "Gamma0_VV.img")
       
       file_VH_out = os.path.join(path3, folder_1[24:32] + "_sVH.tif")
       file_VV_out = os.path.join(path3, folder_1[24:32] + "_sVV.tif")
       file_bVH_out = os.path.join(path3, folder_1[24:32] + "_bVH.tif")
       file_bVV_out = os.path.join(path3, folder_1[24:32] + "_bVV.tif")
       file_gVH_out = os.path.join(path3, folder_1[24:32] + "_gVH.tif")
       file_gVV_out = os.path.join(path3, folder_1[24:32] + "_gVV.tif")
       command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_Superimpose -inr " +  file_inr + " -inm " + file_VV + " -out " +  file_VV_out +" -interpolator linear"
       os.system(command)
       command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_Superimpose -inr " +  file_inr + " -inm " + file_VH + " -out " +  file_VH_out +" -interpolator linear"
       os.system(command)

       command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_Superimpose -inr " +  file_inr + " -inm " + file_bVV + " -out " +  file_bVV_out +" -interpolator linear"
       os.system(command)
       command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_Superimpose -inr " +  file_inr + " -inm " + file_bVH + " -out " +  file_bVH_out +" -interpolator linear"
       os.system(command)

       command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_Superimpose -inr " +  file_inr + " -inm " + file_gVV + " -out " +  file_gVV_out +" -interpolator linear"
       os.system(command)
       command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_Superimpose -inr " +  file_inr + " -inm " + file_gVH + " -out " +  file_gVH_out +" -interpolator linear"
       os.system(command)


# dir_list2 = os.listdir(path)
# dir_list2.sort()
# park = imageio.imread(r"D:\Albufera_2019_processed\subset_albufera\Dataset\park2.tif")

# for folder_2 in dir_list2: 
#     if folder_2.find(".dim") == -1: 
# #        print(folder_2)
#         file_B8 = os.path.join(path, folder_2, "B8.img")
#         file_B4 = os.path.join(path, folder_2, "B4.img")
#         file_B7 = os.path.join(path, folder_2, "B7.img")
#         file_B5 = os.path.join(path, folder_2, "B5.img")
#         file_B11 = os.path.join(path, folder_2, "B11.img")
#         file_B3 = os.path.join(path, folder_2, "B3.img")
#         file_NDVI_out = os.path.join(path3, folder_2[18:26] + "_NDVI_pre.tif")
#         file_NDVI_out2 = os.path.join(path3, folder_2[18:26] + "_NDVI.tif")
#         file_MNDWI_out = os.path.join(path3, folder_2[18:26] + "_MNDWI_pre.tif")
#         file_MNDWI_out2 = os.path.join(path3, folder_2[18:26] + "_MNDWI.tif")
#         file_REVI_out = os.path.join(path3, folder_2[18:26] + "_RE-VI_pre.tif")
#         file_REVI_out2 = os.path.join(path3, folder_2[18:26] + "_RE-VI.tif")
#         file_NDVI_out3 = os.path.join(path3, folder_2[18:26] + "_NDVI_Thre.tif")
#         file_MNDWI_out3 = os.path.join(path3, folder_2[18:26] + "_MNDWI_Thre.tif")

#         file_VV = r"D:\Albufera_2019_processed\subset_albufera\Dataset\20191210_VV.tif"
#         print(folder_2[22:24])
#         if folder_2[22:24] == '12':

# #            command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il " +  file_B8 + " " + file_B4 +  " -out " +  file_NDVI_out +" -exp \"(im1b1 - im2b1 + 10^(-16))/(im1b1 + im2b1 + 10^(-16))\""
# #            print(command)
# #            os.system(command)
# #            command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_Superimpose -inr " +  file_VV + " -inm " + file_NDVI_out + " -out " +  file_NDVI_out2 +" -interpolator linear"
# #            os.system(command)
# #            command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il " +  file_B7 + " " + file_B5 +  " -out " +  file_REVI_out +" -exp \"(im1b1 - im2b1 + 10^(-16))/(im1b1 + im2b1 + 10^(-16))\""
# #            print(command)
# #            os.system(command)
# #            command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_Superimpose -inr " +  file_VV + " -inm " + file_REVI_out + " -out " +  file_REVI_out2 +" -interpolator linear"
# #            os.system(command)
# #            command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il " +  file_B3 + " " + file_B11 +  " -out " +  file_MNDWI_out +" -exp \"(im1b1 - im2b1 + 10^(-16))/(im1b1 + im2b1 + 10^(-16))\""
# #            print(command)
# #            os.system(command)
# #            command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_Superimpose -inr " +  file_VV + " -inm " + file_MNDWI_out + " -out " +  file_MNDWI_out2 +" -interpolator linear"
# #            os.system(command)
            
# #            command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il " + file_MNDWI_out2 +  " -out " +  file_MNDWI_out3 +" -exp \"im1b1 > 0? 1:0\""
# #            print(command)
# #            os.system(command)
            
# #            command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il " + file_NDVI_out2 +  " -out " +  file_NDVI_out3 +" -exp \"im1b1 > 0.7? 1:0\""
# #            print(command)
# #            os.system(command)
# #           
            
#             ndvi = imageio.imread(file_NDVI_out3)
#             mndwi = imageio.imread(file_MNDWI_out3)
#             IMD_TCD_sum = np.sum(ndvi*mndwi*(park/255))
#             print(IMD_TCD_sum)
#             x_mix = np.ndarray(shape=(ndvi.shape[0], ndvi.shape[1], 2), dtype='float32')
#             x_mix[:,:,0] = ndvi
#             x_mix[:,:,1] = mndwi
#             mix_LAYERS = ndvi*mndwi*(park/255)
#             num_pix_Layers = np.sum(mix_LAYERS)
#             mix_LAYERS_2 = np.squeeze(x_mix)
#             mix_LAYERS_2 = np.argmax(mix_LAYERS_2, axis = -1)
#             ndvi_2 = mix_LAYERS_2 == 0 
#             mndwi_2 = mix_LAYERS_2 == 1
#             ndvi1 = ndvi*ndvi_2
#             mndwi1 = mndwi*mndwi_2
#             IMD_TCD_sum = np.sum(ndvi1*mndwi1*(park/255))
#             print(IMD_TCD_sum)
#         else:
#             file_NDVI_out2 = os.path.join(path3, folder_2[18:26] + "_NDVI_Pre.tif")
#             file_MNDWI_out2 = os.path.join(path3, folder_2[18:26] + "_MNDWI_Pre.tif")

# #            command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il " +  file_B8 + " " + file_B4 +  " -out " +  file_NDVI_out +" -exp \"(im1b1 - im2b1 + 10^(-16))/(im1b1 + im2b1 + 10^(-16))\""
# #            print(command)
# #            os.system(command)
# #            command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_Superimpose -inr " +  file_inr + " -inm " + file_NDVI_out + " -out " +  file_NDVI_out2 +" -interpolator linear"
# #            os.system(command)
# #            command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il " +  file_B7 + " " + file_B5 +  " -out " +  file_REVI_out +" -exp \"(im1b1 - im2b1 + 10^(-16))/(im1b1 + im2b1 + 10^(-16))\""
# #            print(command)
# #            os.system(command)
# #            command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_Superimpose -inr " +  file_inr + " -inm " + file_REVI_out + " -out " +  file_REVI_out2 +" -interpolator linear"
# #            os.system(command)
# #            command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il " +  file_B3 + " " + file_B11 +  " -out " +  file_MNDWI_out +" -exp \"(im1b1 - im2b1 + 10^(-16))/(im1b1 + im2b1 + 10^(-16))\""
# #            print(command)
# #            os.system(command)
# #            command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_Superimpose -inr " +  file_inr + " -inm " + file_MNDWI_out + " -out " +  file_MNDWI_out2 +" -interpolator linear"
# #            os.system(command)
# ##        
# #            command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il " + file_MNDWI_out2 +  " -out " +  file_MNDWI_out3 +" -exp \"im1b1 > 0? 1:0\""
# #            print(command)
# #            os.system(command)
            
# #            command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il " + file_NDVI_out2 +  " -out " +  file_NDVI_out3 +" -exp \"im1b1 > 0.7? 1:0\""
# #            print(command)
# #            os.system(command)
#             ndvi = imageio.imread(file_NDVI_out3)
#             mndwi = imageio.imread(file_MNDWI_out3)
#             IMD_TCD_sum = np.sum(ndvi*mndwi*(park/255))
#             print(IMD_TCD_sum)
#             x_mix = np.ndarray(shape=(ndvi.shape[0], ndvi.shape[1], 2), dtype='float32')
#             x_mix[:,:,0] = ndvi
#             x_mix[:,:,1] = mndwi
#             mix_LAYERS = ndvi*mndwi*(park/255)
#             num_pix_Layers = np.sum(mix_LAYERS)
#             mix_LAYERS_2 = np.squeeze(x_mix)
#             mix_LAYERS_2 = np.argmax(mix_LAYERS_2, axis = -1)
#             ndvi_2 = mix_LAYERS_2 == 0 
#             mndwi_2 = mix_LAYERS_2 == 1
#             ndvi1 = ndvi*ndvi_2
#             mndwi1 = mndwi*mndwi_2
#             IMD_TCD_sum = np.sum(ndvi1*mndwi1*(park/255))
#             print(IMD_TCD_sum)

#         mndwi_file = os.path.join(path3, folder_2[18:26] + "_MNDWI_2.tif")
#         mndwi1 = mndwi1.astype('float32')
#         imsave(mndwi_file, mndwi1)

#         ndvi_file = os.path.join(path3, folder_2[18:26] + "_NDVI_2.tif")
#         ndvi1 = ndvi1.astype('float32')
#         imsave(ndvi_file, ndvi1)
