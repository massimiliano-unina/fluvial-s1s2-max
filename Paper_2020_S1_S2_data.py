# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:19:46 2020

@author: massi
"""

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
path = r"D:\Albufera_2019_processed\new_S2_S1\\"
path2 = r"D:\Albufera_2019_processed\subset_albufera\subset_albufera\s2\\"
path3 = r"D:\Albufera_2019_processed\Dataset_paper_2020\\"

if not os.path.exists(path3):
    os.makedirs(path3)
    
dir_list = os.listdir(path)
dir_list.sort()

dir_list2 = os.listdir(path2)
dir_list2.sort()

dir_list3 = os.listdir(path3)
dir_list3.sort()

#file_inr = r"D:\Albufera_2019_processed\subset_albufera\Dataset\20190506_NDVI_pre.tif"

dic_month = {"Jan": "01", "Feb" : "02", "Mar" : "03", "Apr" : "04", "May":"05", "Jun" : "06", "Jul" : "07", "Aug" : "08", "Sep" : "09", "Oct": "10", "Nov" : "11", "Dec": "12"}

#for file_1 in dir_list: 
#   file_2 = file_1[-10:-6] + dic_month[file_1[-13:-10]] + file_1[-15:-13]
#   file_VV_out = os.path.join(path3, file_2 + "_"+  file_1[7:9] + ".tif")
#   file_VV = os.path.join(path, file_1)
#   command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il " +  file_VV + " -out " +  file_VV_out +" -exp \"im1b1\""       
#   os.system(command)

#for folder_1 in dir_list2: 
#   if folder_1.find(".dim") == -1: 
#       dir_list_inside = os.listdir(os.path.join(path2,folder_1))
#       for file_1 in dir_list_inside:
#           if file_1.find("B3") != -1:# or file_1.find("B11") != -1: # file_1.find("B4") != -1 or file_1.find("B8") != -1 or file_1.find("B11") != -1:
#               file_VV_out = os.path.join(path3,folder_1[18:26] + "_" + file_1[:-4] + ".tif")
#               file_VV = os.path.join(path2, folder_1, file_1)
#               command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il " +  file_VV + " -out " +  file_VV_out +" -exp \"im1b1\""       
#               os.system(command)

date = ["20190121",
        "20190225",
        "20190317",
        "20190411",
        "20190506",
        "20190610",
        "20190725",
        "20190824",
        "20190918",
        "20191008",
        "20191112",
        "20191207"]


for data in date: 
#    B3 = os.path.join(path3, data + "_B3.tif")
#    B4 = os.path.join(path3, data + "_B4.tif")
#    B8 = os.path.join(path3, data + "_B8.tif")
#    B11 = os.path.join(path3, data + "_B11.tif")     
    NDVI = os.path.join(path3, data + "_NDVI_pre.tif")     
    MNDWI = os.path.join(path3, data + "_MNDWI_pre.tif")     
    NDBI = os.path.join(path3, data + "_NDBI_pre.tif")     
#    command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il " +  B8 + " " + B11 + " -out " +  NDWI +" -exp \"(im1b1 - im2b1)/(im1b1 + im2b1)\""       
#    os.system(command)
#    command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il " +  B8 + " " + B4 + " -out " +  NDVI +" -exp \"(im1b1 - im2b1)/(im1b1 + im2b1)\""       
#    os.system(command)
#    command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il " +  B11 + " " + B3 + " -out " +  MNDWI +" -exp \"(im1b1 - im2b1)/(im1b1 + im2b1)\""       
#    os.system(command)
    command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il " +  MNDWI + " " + NDVI + " -out " +  NDBI +" -exp \"(im1b1 - im2b1)\""       
    os.system(command)
    