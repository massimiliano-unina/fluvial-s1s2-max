# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 12:33:46 2020

@author: massi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:02:43 2020

@author: massi
"""
import imageio 
from matplotlib import pyplot as plt
import numpy as np 
from tifffile import imsave 
import random
import os 
from scipy import ndimage, misc

#############
folder = r"D:\Works\08_month\\"
dir_list = os.listdir(folder)
dir_list.sort()

#N_1 = {"139": 3,"168": 4 }

N_1 = {"168": 3}
print(dir_list)
N = 6
Out = 3
num = 1

ps1 = 128
r1 = 128


ps = 128
r = 32

enlarge = 64


rotation = [0,45,90, 135, 180, 225, 270, 315]
rotation1 = [0,45,90]
rotation2 = [0] #[0, 135]

A = len(rotation)
B = len(rotation1)
C = len(rotation2)

x_train = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
y_train = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
x_val = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
y_val = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
x_test = np.ndarray(shape=(0, ps1, ps1, N), dtype='float32')
y_test = np.ndarray(shape=(0, ps1,ps1, Out), dtype='float32')
features = ["Germany_Aug_2018", "corine_u8_c.", "geo_localthetainc.","geo_mean_gamma0dB.","geo_mean_rho6.","geo_rhoLT.","geo_tau."]

patches_iniziali = 0 
patches_finali = 0
for n_1 in N_1: 
    for k1 in range(N_1[n_1]): 
        folder_1 = folder + str(n_1) + "_orbit\TS_" + str(k1 + 1) + "\\channel_vv\\"
        for feature in features: 
            if feature == "Germany_Aug_2018":
                feature_1 = feature + "_"+ str(n_1) + "orbit_TS" + str(k1 + 1) + "_resampled."
                file_inp =  folder_1 + feature_1 + "data\\band_1.img"
                file_out = folder_1 + "NDVI.tif"
            else: 
                file_inp =  folder_1 + feature + "data\\band_1_S.img"
                file_out = folder_1 + feature + "tif"
            command = r'C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il "' + file_inp + '" -out "' + file_out +'" -exp im1b1'
            print(command)
            os.system(command) 