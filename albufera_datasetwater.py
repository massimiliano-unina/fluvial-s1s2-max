import os 
import numpy as np 
import imageio 
from matplotlib import pyplot as plt 
from tifffile import imsave
path = r"D:\Albufera_2019_processed\subset_albufera\s2\\"
path2 = r"C:\Users\massi\Downloads\sentinel1_Salerno\Albufera-20200415T192751Z-001\Albufera\File_Training\Input\2018\\"  #"D:\Albufera_2019_processed\subset_albufera\s1\\"
path3 = r"C:\Users\massi\Downloads\sentinel1_Salerno\Albufera-20200415T192751Z-001\Albufera\File_Training\Input\2018\processed\\"

if not os.path.exists(path3):
    os.makedirs(path3)
    
dir_list = os.listdir(path2)
dir_list.sort()

file_inr = r"C:\Users\massi\Downloads\sentinel1_Salerno\Albufera-20200415T192751Z-001\Albufera\File_Training\Input\2018\S_P_FV_PV_Condition_15022018.tif"
# file_inr = r"C:\Users\massi\Downloads\sentinel1_Salerno\Albufera-20200415T192751Z-001\Albufera\File_Training\Input\2020\S_P_FV_PV_20022020_.tif"
cc = 0
for file_1 in dir_list: 
    if file_1.find("Sigma0") != -1:
        file_VH = os.path.join(path2, file_1)
        file_VH_out = os.path.join(path3,file_1)
        command = r"C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_Superimpose -inr " +  file_inr + " -inm " + file_VH + " -out " +  file_VH_out +" -interpolator linear"
        os.system(command)
