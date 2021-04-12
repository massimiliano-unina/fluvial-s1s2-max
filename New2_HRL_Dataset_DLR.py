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
folder = r"D:\08_month_40m\\"
folder_out = r"D:\German_Train_Naples\\"

if not os.path.exists(folder_out):
    os.makedirs(folder_out)

dir_list = os.listdir(folder)
dir_list.sort()

N_1 = {"139": 3,"168": 4 }
# N_1 = {"168":1}
#N_1 = {"168": 4}
print(dir_list)
N = 6
Out = 3
num = 1

ps1 = 128
r1 = 128


ps = 128
r = 128
enlarge = 64


rotation = [0,45,90, 135, 180, 225, 270, 315]
rotation1 = [0,45,90]
rotation2 = [0] #[0, 135]

A = len(rotation)
B = len(rotation1)
C = len(rotation2)


features = ["geo_ndvi - Copia.","geo_ndvi.", "geo_localthetainc.","geo_mean_gamma0_dB.","geo_mean_rho6.","geo_rhoLT.","geo_tau.", "IMD_2015_020m_eu_03035_d05_Merge_wgs84", "TCD_2015_020m_eu_03035_d05_Merge_wgs84","WAW_2015_020m_eu_03035_d06_Merge_wgs84"]

patches_iniziali = 0 
patches_finali = 0
final_out = 0 
final_test = 0
for n_1 in N_1: 
    for k1 in range(N_1[n_1]):
        # if n_1 ==  "168": 
        #     k1 = k1 + 1
        folder_1 = folder + str(n_1) + "_orbit\TS_" + str(k1) + "\\channel_vv_hrl_2015\posting_40m\\"
        folder_2 = folder + str(n_1) + "_orbit\TS_" + str(k1) + "\\hrl_2015\\"
        folder_out_2 = folder_out # + str(n_1) + "_orbit\TS_" + str(k1) + "\\channel_vv_100m\\"
        
        if not os.path.exists(folder_out_2):
            os.makedirs(folder_out_2)
    
        
        for feature in features: 
            # print(feature)
            if feature.find(".") != -1:
                file_out = folder_1 + feature + "tiff"
            else: 
                file_out = folder_2 + feature + ".tif"
            print(file_out)
            add = imageio.imread(file_out)
            if feature == "geo_ndvi.":
                ndvi = np.asarray(add)
            elif feature == "geo_ndvi - Copia.":
                corine = np.asarray(add)
            elif feature == "geo_localthetainc.":
                localthetainc = np.asarray(add)
            elif feature == "geo_mean_gamma0_dB.":
                gamma_0 = np.asarray(add)
            elif feature== "geo_mean_rho6.":
                rho_6 = np.asarray(add)
            elif feature == "geo_rhoLT.":
                rhoLT = np.asarray(add)
            elif feature == "geo_tau.":
                tau = np.asarray(add)
            elif feature == "IMD_2015_020m_eu_03035_d05_Merge_wgs84":
                hrl_ARTIFICIAL_SURFACES = np.asarray(add)
            elif feature == "WAW_2015_020m_eu_03035_d06_Merge_wgs84":
                hrl_WATER = np.asarray(add)
            elif feature == "TCD_2015_020m_eu_03035_d05_Merge_wgs84":
                hrl_FOREST = np.asarray(add)
            # elif feature == "NTCD_Thre.":
            #     hrl_NOFOREST = np.asarray(add)
            #     print(feature)
        size_ndvi = rhoLT.shape

        # mask_rhoLT = (rhoLT == 0 ) + (hrl_WATER)
        [s1, s2] = hrl_WATER.shape
        # hrl_WATER = (hrl_WATER == 1)*(hrl_WATER == 255)
        # hrl_ARTIFICIAL_SURFACES = (hrl_ARTIFICIAL_SURFACES > 50)*np.invert(hrl_WATER)
        # hrl_FOREST = (hrl_FOREST > 50)*np.invert(hrl_WATER)*np.invert(hrl_ARTIFICIAL_SURFACES)
        # hrl_NOFOREST = np.invert(hrl_WATER)*np.invert(hrl_ARTIFICIAL_SURFACES)*np.invert(hrl_FOREST)

        hrl_WATER = (corine == 0)
        hrl_ARTIFICIAL_SURFACES = (corine == 45)
        hrl_FOREST = (corine == 130)
        hrl_NOFOREST = (corine == 215)

        print(np.sum(hrl_WATER*hrl_NOFOREST*hrl_FOREST*hrl_ARTIFICIAL_SURFACES))
        if (n_1 ==  "168" and k1 != 0) or (n_1 == "139"):
            print("training")
            print(n_1 + "k1 : " + str(k1))
            p2 = []
            # print(len(p2))
            for y in range(500,s1-500-ps+1,r): 
                for x in range(500,s2-500-ps+1,r):
                    mask_d0 = hrl_WATER[y:y+ps,x:x+ps]
                    mask_d0_corine_ARTIFICIAL_SURFACES= hrl_ARTIFICIAL_SURFACES[y:y+ps,x:x+ps]
                    mask_d0_corine_FOREST = hrl_FOREST[y:y+ps,x:x+ps]
                    mask_d0_corine_NOFOREST = hrl_NOFOREST[y:y+ps,x:x+ps]
                    [m1,m2] = mask_d0.shape
                    s_0 =  mask_d0.sum()
                    s_123 = [mask_d0_corine_ARTIFICIAL_SURFACES.sum(), mask_d0_corine_FOREST.sum(), mask_d0_corine_NOFOREST.sum() ]
                    materials = np.where(s_123 == np.max(s_123))[0][0]
                    if s_0 == 0:
                        p2.append([y,x,materials])
            p_train = []
            p_val = []
            
            P1 = len(p2)
            p = p2#[p2[s] for s in v]  
            random.shuffle(p)
            # patches_finali += P
            x_train_k = np.ndarray(shape=(ps, ps, N), dtype='float32')
            y_train_k = np.ndarray(shape=(ps, ps, Out), dtype='float32')
    
            n = 0
            for patch in p:
                y0, x0 = patch[0], patch[1]
    
                x_train_k[:,:,0] = gamma_0[y0:y0+ps,x0:x0+ps]
                x_train_k[:,:,1] = rhoLT[y0:y0+ps,x0:x0+ps]
                x_train_k[:,:,2] = tau[y0:y0+ps,x0:x0+ps]
                x_train_k[:,:,3] = localthetainc[y0:y0+ps,x0:x0+ps]
                x_train_k[:,:,4] = rho_6[y0:y0+ps,x0:x0+ps]
                x_train_k[:,:,5] = ndvi[y0:y0+ps,x0:x0+ps]
                
                y_train_k[:,:,0]= hrl_ARTIFICIAL_SURFACES[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_train_k[:,:,1] = hrl_FOREST[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_train_k[:,:,2] = hrl_NOFOREST[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                np.save(os.path.join(folder_out_2,'X_train_' + str(final_out) + '.npy'),x_train_k)
                np.save(os.path.join(folder_out_2,'Y_train_' + str(final_out) + '.npy'),y_train_k)

                # imsave(os.path.join(folder_out_2,'Y_train_' + str(final_out) + '.tif'),y_train_k)
                # imsave(os.path.join(folder_out_2,'X_train_' + str(final_out) + '.tif'),x_train_k)
                n = n + 1
                final_out += 1
        else: 
            print("testing")
            print(n_1 + "k1 : " + str(k1))
            p2 = []
            for y in range(1,s1-ps1+1,r1): 
                for x in range(1,s2-ps1+1,r1):
                    mask_d0 = hrl_WATER[y:y+ps1,x:x+ps1]
                    mask_d0_corine_ARTIFICIAL_SURFACES= hrl_ARTIFICIAL_SURFACES[y:y+ps1,x:x+ps1]
                    mask_d0_corine_FOREST = hrl_FOREST[y:y+ps1,x:x+ps1]
                    mask_d0_corine_NOFOREST = hrl_NOFOREST[y:y+ps1,x:x+ps1]
                    [m1,m2] = mask_d0.shape
                    s_0 =  mask_d0.sum()
                    s_123 = [mask_d0_corine_ARTIFICIAL_SURFACES.sum(), mask_d0_corine_FOREST.sum(), mask_d0_corine_NOFOREST.sum() ]
                    materials = np.where(s_123 == np.max(s_123))[0][0]
                    if s_0 == 0:
                        p2.append([y,x,materials])
            p_test = p2
            
            P1 = len(p_test)
            # p = p2#[p2[s] for s in v]  
#            random.shuffle(p)
            P = len(p_test)
            patches_finali += P
            x_test_k = np.ndarray(shape=(ps1, ps1, N), dtype='float32')
            y_test_k = np.ndarray(shape=(ps1, ps1, Out), dtype='float32')
    
            n1 = 0
            for patch in p_test:
                y0, x0 = patch[0], patch[1]
                x_test_k[:,:,0] = gamma_0[y0:y0+ps,x0:x0+ps]
                x_test_k[:,:,1] = rhoLT[y0:y0+ps,x0:x0+ps]
                x_test_k[:,:,2] = tau[y0:y0+ps,x0:x0+ps]
                x_test_k[:,:,3] = localthetainc[y0:y0+ps,x0:x0+ps]
                x_test_k[:,:,4] = rho_6[y0:y0+ps,x0:x0+ps]
                x_test_k[:,:,5] = ndvi[y0:y0+ps,x0:x0+ps]
                
                y_test_k[:,:,0]= hrl_ARTIFICIAL_SURFACES[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_test_k[:,:,1] = hrl_FOREST[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_test_k[:,:,2] = hrl_NOFOREST[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                # imsave(os.path.join(folder_out_2,'X_test_' + str(final_test) + '.tif'),x_test_k)
                # imsave(os.path.join(folder_out_2,'Y_test_' + str(final_test) + '.tif'),y_test_k)


                np.save(os.path.join(folder_out_2,'X_test_' + str(final_test) + '.npy'),x_test_k)
                np.save(os.path.join(folder_out_2,'Y_test_' + str(final_test) + '.npy'),y_test_k)
                final_test += 1
                n1 = n1 + 1
        
            num +=1


train_val_p = n
test_p = n1

folder_out_2 = r"D:\German_Indices\\"   

if not os.path.exists(folder_out_2):
    os.makedirs(folder_out_2)


ind = np.arange(train_val_p)
np.random.shuffle(ind)
train_perc = 0.9
train_samp = int(train_val_p*train_perc)
np.save(os.path.join(folder_out_2, 'train_ind.npy'),ind[:train_samp])
np.save(os.path.join(folder_out_2, 'val_ind.npy'),ind[train_samp:])