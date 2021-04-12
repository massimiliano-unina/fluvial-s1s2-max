# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:07:39 2020

@author: massi
"""
from __future__ import print_function

import sys
import os
from tifffile import imsave
import numpy as np
#import gdal
import imageio
from scipy.misc import imresize
from scipy.ndimage import gaussian_filter, morphology,generate_binary_structure

import gdal
    #############
    # short names

path = r"D:\Albufera_2019_processed\subset_albufera\Dataset\\"
folder_out_2 = r"D:\Albufera_2019_processed\Training\Beta\\"

if not os.path.exists(folder_out_2):
    os.makedirs(folder_out_2)


#############
dir_list = os.listdir(path)
dir_list.sort()
print(dir_list)
ps = 128 

N = 6
Out = 5
import random
# x_train = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
# x_btrain = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
# x_gtrain = np.ndarray(shape=(0, ps, ps, N), dtype='float32')

# y_train = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')

# x_val = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
# x_bval = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
# x_gval = np.ndarray(shape=(0, ps, ps, N), dtype='float32')

# y_val = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')

# files_VH = []
# files_VV = []
# files_gVH = []
# files_gVV = []
# files_bVH = []
# files_bVV = []

# files_NDVI = []
# files_MNDWI = []
# files_NDVIp = []
# files_MNDWIp = []
# for file in dir_list: 
#     if file.find("201901") == -1 and file.find("201907") == -1 :  ############# 
#         if file.find("sVV") != -1: 
#             files_VV.append(file)
#         if file.find("sVH") != -1: 
#             files_VH.append(file)
#         if file.find("bVV") != -1: 
#             files_gVV.append(file)
#         if file.find("bVH") != -1: 
#             files_gVH.append(file)
#         if file.find("bVV") != -1: 
#             files_bVV.append(file)
#         if file.find("bVH") != -1: 
#             files_bVH.append(file)
            
#         if file.find("NDVI_2") != -1: 
#             files_NDVI.append(file)
#         if file.find("MNDWI_2") != -1: 
#             files_MNDWI.append(file)
#         if file.find("NDVI_pre") != -1: 
#             files_NDVIp.append(file)
#         if file.find("MNDWI_pre") != -1: 
#             files_MNDWIp.append(file)

# print(len(files_VV))
# print(len(files_VH))
# print(len(files_gVV))
# print(len(files_gVH))
# print(len(files_bVV))
# print(len(files_bVH))
# print(len(files_NDVI))
# print(len(files_MNDWI))
# print(len(files_NDVIp))
# print(len(files_MNDWIp))

park_file = "park2.tif"
dataset = gdal.Open(path + park_file, gdal.GA_ReadOnly)
park2 = dataset.ReadAsArray()
dataset = None
park2 = park2/255
park = 1 - park2

[s1, s2] = park.shape

# veg = 0 
# wat = 0 
# noveg = 0
# r = 32
# mndwi2 = np.zeros((park2.shape[0], park2.shape[1]))
# final_out = 0
# final_out_2 = 0 

# for MNDWI_ind in range(len(files_MNDWI)): 
#     file_MNDWI = os.path.join(path, files_MNDWI[MNDWI_ind])
#     file_NDVI = os.path.join(path, files_NDVI[MNDWI_ind])
#     file_MNDWIp = os.path.join(path, files_MNDWIp[MNDWI_ind])
#     file_NDVIp = os.path.join(path, files_NDVIp[MNDWI_ind])

#     dataset = gdal.Open(file_MNDWI, gdal.GA_ReadOnly)
#     mndwi = dataset.ReadAsArray()
#     dataset = None
#     dataset = gdal.Open(file_NDVI, gdal.GA_ReadOnly)
#     ndvi = dataset.ReadAsArray()
#     dataset = None
#     dataset = gdal.Open(file_MNDWIp, gdal.GA_ReadOnly)
#     mndwip = dataset.ReadAsArray()
#     dataset = None
#     dataset = gdal.Open(file_NDVIp, gdal.GA_ReadOnly)
#     ndvip = dataset.ReadAsArray()
#     dataset = None

#     file_VV_0 = os.path.join(path, files_gVV[3*MNDWI_ind])
#     file_VV_1 = os.path.join(path, files_gVV[3*MNDWI_ind + 1])
#     file_VV_2 = os.path.join(path, files_gVV[3*MNDWI_ind + 2])
#     file_VH_0 = os.path.join(path, files_gVH[3*MNDWI_ind])
#     file_VH_1 = os.path.join(path, files_gVH[3*MNDWI_ind + 1])
#     file_VH_2 = os.path.join(path, files_gVH[3*MNDWI_ind + 2])
#     dataset = gdal.Open(file_VV_0, gdal.GA_ReadOnly)
#     gvv_0 = dataset.ReadAsArray()
#     dataset = None
#     dataset = gdal.Open(file_VV_1, gdal.GA_ReadOnly)
#     gvv_1 = dataset.ReadAsArray()
#     dataset = None
#     dataset = gdal.Open(file_VV_2, gdal.GA_ReadOnly)
#     gvv_2 = dataset.ReadAsArray()
#     dataset = None
#     dataset = gdal.Open(file_VH_0, gdal.GA_ReadOnly)
#     gvh_0 = dataset.ReadAsArray()
#     dataset = None
#     dataset = gdal.Open(file_VH_1, gdal.GA_ReadOnly)
#     gvh_1 = dataset.ReadAsArray()
#     dataset = None
#     dataset = gdal.Open(file_VH_2, gdal.GA_ReadOnly)
#     gvh_2 = dataset.ReadAsArray()
#     dataset = None


#     cndvi = (1 - ndvi)*(1 - mndwi)

# #    mndwi2 += mndwi
# #
# ##    NOVEG = files_MNDWI[MNDWI_ind]
# ##    nome_file_Noveg = os.path.join(path, NOVEG[:9] + "NoVeg.tif" )
# ##    cndvi = cndvi.astype('float32')
# ##    imsave(nome_file_Noveg, cndvi)
# #    veg += (np.sum(ndvi*park2))
# #    noveg += (np.sum(cndvi*park2))
# #    wat += (np.sum(mndwi*park2))
# #
# #nome_file_Wat = os.path.join(path, "Water_2019.tif" )
# #cndvi = mndwi2.astype('float32')
# #imsave(nome_file_Wat, cndvi)
# #
# #
# #tot = veg + noveg + wat
# #print("vegetation : " + str((veg/tot)*100) + "%" )
# #print("no-vegetation : " + str((noveg/tot)*100)+ "%" )
# #print("water : " + str((wat/tot)*100)+ "%" )


# #    #### other additional input   
#     if file_MNDWI.find("201907") != -1 or file_MNDWI.find("201908") != -1 or file_MNDWI.find("201909") != -1: 
#         r = 16
#     else:
#         r = 32
#     [s1, s2] = gvv_1.shape
#     p2 = []
#     print(len(p2))
#     for y in range(1,s1-ps+1,r): 
#         for x in range(1,s2-ps+1,r):
#             mask_d0 = park[y:y+ps,x:x+ps]
#             [m1,m2] = mask_d0.shape
#             s_0 =  mask_d0.sum()
#             if s_0 == 0: 
#                 p2.append([y,x])
#     p_train = []
#     p_val = []
#     print(len(p2))
#     P = len(p2)# 3000#int(3000)
#     p = p2#[p2[s] for s in v]  
#     random.shuffle(p)
#     p_train, p_val= p[:int(0.95*P)],p[int(0.95*P):P]
#     print(len(p_train))
#     print(len(p_val))
    
#     y_train_k = np.ndarray(shape=(ps, ps, Out), dtype='float32')

#     x_gtrain_k = np.ndarray(shape=( ps, ps, N), dtype='float32')

    
#     # n = 0
#     for patch in p_train:
#         y0, x0 = patch[0], patch[1]
#         x_gtrain_k[ :,:,0] = gvv_0[y0:y0+ps,x0:x0+ps]
#         x_gtrain_k[ :,:,1] = gvv_1[y0:y0+ps,x0:x0+ps]
#         x_gtrain_k[ :,:,2] = gvv_2[y0:y0+ps,x0:x0+ps]
#         x_gtrain_k[ :,:,3] = gvh_0[y0:y0+ps,x0:x0+ps]
#         x_gtrain_k[ :,:,4] = gvh_1[y0:y0+ps,x0:x0+ps]
#         x_gtrain_k[ :,:,5] = gvh_2[y0:y0+ps,x0:x0+ps]

#         y_train_k[ :,:,0]= cndvi[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#         y_train_k[ :,:,1] = ndvi[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#         y_train_k[ :,:,2] = mndwi[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#         y_train_k[ :,:,3] = ndvip[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#         y_train_k[ :,:,4] = mndwip[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#         np.save(os.path.join(folder_out_2,'X_train_' + str(final_out) + '.npy'),x_gtrain_k)
#         np.save(os.path.join(folder_out_2,'Y_train_' + str(final_out) + '.npy'),y_train_k)
#         # imsave(os.path.join(folder_out_2,'Y_train_' + str(final_out) + '.tif'),y_train_k)
#         # imsave(os.path.join(folder_out_2,'X_train_' + str(final_out) + '.tif'),x_gtrain_k)
#         final_out += 1 
#     #     n = n + 1
#     # y_train = np.concatenate((y_trai  y_train_k))

#     # x_gtrain = np.concatenate((x_gtrai  x_gtrain_k))


#     #            x_train2 = np.concatenate((x_train2, x_train_k2))
#     #            y_train2 = np.concatenate((y_train2, y_train_k2))
    
    
#     x_val_k = np.ndarray(shape=( ps, ps, N), dtype='float32')
#     y_val_k = np.ndarray(shape=( ps, ps, Out), dtype='float32')

#     x_gval_k = np.ndarray(shape=( ps, ps, N), dtype='float32')

#     x_bval_k = np.ndarray(shape=( ps, ps, N), dtype='float32')


#     # n = 0
#     for patch in p_val:
#         y0, x0 = patch[0], patch[1]


#         x_gval_k[ :,:,0] = gvv_0[y0:y0+ps,x0:x0+ps]
#         x_gval_k[ :,:,1] = gvv_1[y0:y0+ps,x0:x0+ps]
#         x_gval_k[ :,:,2] = gvv_2[y0:y0+ps,x0:x0+ps]
#         x_gval_k[ :,:,3] = gvh_0[y0:y0+ps,x0:x0+ps]
#         x_gval_k[ :,:,4] = gvh_1[y0:y0+ps,x0:x0+ps]
#         x_gval_k[ :,:,5] = gvh_2[y0:y0+ps,x0:x0+ps]

        
#         y_val_k[ :,:,0]= cndvi[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#         y_val_k[ :,:,1] = ndvi[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#         y_val_k[ :,:,2] = mndwi[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#         y_val_k[ :,:,1] = ndvip[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#         y_val_k[ :,:,2] = mndwip[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#         np.save(os.path.join(folder_out_2,'X_val_' + str(final_out) + '.npy'),x_gval_k)
#         np.save(os.path.join(folder_out_2,'Y_val_' + str(final_out) + '.npy'),y_val_k)
        
#         # imsave(os.path.join(folder_out_2,'Y_train_' + str(final_out_2) + '.tif'),y_val_k)
#         # imsave(os.path.join(folder_out_2,'X_train_' + str(final_out_2) + '.tif'),x_gval_k)
#         final_out_2 += 1 
        # n = n + 1
    # y_val = np.concatenate((y_val, y_val_k))

    # x_gval = np.concatenate((x_gval, x_gval_k))


y_test = np.ndarray(shape=(2, s1, s2, Out), dtype='float32')

x_gtest = np.ndarray(shape=(2, s1, s2, N), dtype='float32')



files_VH = []
files_VV = []
files_gVH = []
files_gVV = []
files_bVH = []
files_bVV = []

files_NDVI = []
files_MNDWI = []
files_NDVIp = []
files_MNDWIp = []

for file in dir_list: 
    if file.find("201901") != -1 or file.find("201907") != -1 : 
        if file.find("sVV") != -1: 
            files_VV.append(file)
        if file.find("sVH") != -1: 
            files_VH.append(file)
        if file.find("gVV") != -1: 
            files_gVV.append(file)
        if file.find("gVH") != -1: 
            files_gVH.append(file)
        if file.find("bVV") != -1: 
            files_bVV.append(file)
        if file.find("bVH") != -1: 
            files_bVH.append(file)
            
        if file.find("NDVI_2") != -1: 
            files_NDVI.append(file)
        if file.find("MNDWI_2") != -1: 
            files_MNDWI.append(file)
        if file.find("NDVI_pre") != -1: 
            files_NDVIp.append(file)
        if file.find("MNDWI_pre") != -1: 
            files_MNDWIp.append(file)
print(len(files_VV))
print(len(files_VH))
print(len(files_gVV))
print(len(files_gVH))
print(len(files_bVV))
print(len(files_bVH))
print(len(files_NDVI))
print(len(files_MNDWI))
print(len(files_NDVIp))
print(len(files_MNDWIp))

n = 0

for MNDWI_ind in range(len(files_MNDWI)): 
    file_MNDWI = os.path.join(path, files_MNDWI[MNDWI_ind])
    file_NDVI = os.path.join(path, files_NDVI[MNDWI_ind])
    file_MNDWIp = os.path.join(path, files_MNDWIp[MNDWI_ind])
    file_NDVIp = os.path.join(path, files_NDVIp[MNDWI_ind])
    


    file_VV_0 = os.path.join(path, files_gVV[3*MNDWI_ind])
    file_VV_1 = os.path.join(path, files_gVV[3*MNDWI_ind + 1])
    file_VV_2 = os.path.join(path, files_gVV[3*MNDWI_ind + 2])
    file_VH_0 = os.path.join(path, files_gVH[3*MNDWI_ind])
    file_VH_1 = os.path.join(path, files_gVH[3*MNDWI_ind + 1])
    file_VH_2 = os.path.join(path, files_gVH[3*MNDWI_ind + 2])
    dataset = gdal.Open(file_VV_0, gdal.GA_ReadOnly)
    gvv_0 = dataset.ReadAsArray()
    dataset = None
    dataset = gdal.Open(file_VV_1, gdal.GA_ReadOnly)
    gvv_1 = dataset.ReadAsArray()
    dataset = None
    dataset = gdal.Open(file_VV_2, gdal.GA_ReadOnly)
    gvv_2 = dataset.ReadAsArray()
    dataset = None
    dataset = gdal.Open(file_VH_0, gdal.GA_ReadOnly)
    gvh_0 = dataset.ReadAsArray()
    dataset = None
    dataset = gdal.Open(file_VH_1, gdal.GA_ReadOnly)
    gvh_1 = dataset.ReadAsArray()
    dataset = None
    dataset = gdal.Open(file_VH_2, gdal.GA_ReadOnly)
    gvh_2 = dataset.ReadAsArray()
    dataset = None

    dataset = gdal.Open(file_MNDWI, gdal.GA_ReadOnly)
    mndwi = dataset.ReadAsArray()
    dataset = None
    dataset = gdal.Open(file_NDVI, gdal.GA_ReadOnly)
    ndvi = dataset.ReadAsArray()
    dataset = None
    cndvi = (1 - ndvi)*(1 - mndwi)
    dataset = gdal.Open(file_MNDWIp, gdal.GA_ReadOnly)
    mndwip = dataset.ReadAsArray()
    dataset = None
    dataset = gdal.Open(file_NDVIp, gdal.GA_ReadOnly)
    ndvip = dataset.ReadAsArray()
    dataset = None
   


    x_gtest[n, :,:,0] = gvv_0 
    x_gtest[n, :,:,1] = gvv_1 
    x_gtest[n, :,:,2] = gvv_2 
    x_gtest[n, :,:,3] = gvh_0 
    x_gtest[n, :,:,4] = gvh_1 
    x_gtest[n, :,:,5] = gvh_2 


    y_test[n, :,:,0]= cndvi#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    y_test[n, :,:,1] = ndvi#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    y_test[n, :,:,2] = mndwi#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    y_test[n, :,:,3] = ndvip#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    y_test[n, :,:,4] = mndwip#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    n += 1
np.savez("test_data_SAR_Iodice_beta_with_summer_augmentation.npz",x_gtest = x_gtest,y_test = y_test)#,  x_gtrain = x_gtrai y_train = y_trai  x_gval = x_gval, y_val = y_val)    
