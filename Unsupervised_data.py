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

data_path = r"D:\fiumi unsupervised\Dataset_Unsupervised_2021\\" #D:\Albufera_2019_processed\subset_albufera\Dataset\\"
folder_out_2 = r"D:\fiumi unsupervised\Training_Set_Unsupervised\\"

if not os.path.exists(folder_out_2):
    os.makedirs(folder_out_2)

#############
dir_list = os.listdir(data_path)
dir_list.sort()
print(dir_list)
ps = 128 

N = 3
Out = 5
import random

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

# final_out = 0
# final_out_2 = 0 

# for file in dir_list: 
#     if file.find("201911") == -1 and file.find("201912") == -1: # file.find("201901") == -1 and file.find("201907") == -1 :  ############# 
#         if file.find("VV") != -1: 
#             files_gVV.append(file)
#         if file.find("VH") != -1: 
#             files_gVH.append(file)
#         if file.find("NDVI_pre") != -1: 
#             files_NDVIp.append(file)
#         if file.find("MNDWI_pre") != -1: 
#             files_MNDWIp.append(file)

# print(len(files_gVV))
# print(len(files_gVH))
# print(len(files_NDVIp))
# print(len(files_MNDWIp))

# park_file = "park2.tif"
# dataset = gdal.Open(path + park_file, gdal.GA_ReadOnly)
# park2 = dataset.ReadAsArray()
# dataset = None
# park2 = park2/255
# park = 1 - park2

# [s1,s2] = park.shape


y_test = np.ndarray(shape=(14, 256, 128, Out), dtype='float32')
x_test = np.ndarray(shape=(14, 256, 128, N), dtype='float32')

# print(y_test.shape)
veg = 0 
wat = 0 
noveg = 0
r = 32
n = 0
final_out = 0
final_out_2 = 0
n1 = 0 
for file in dir_list: 
   if file.find("B4") != -1 and (file.find("Enza") != -1 or file.find("canneto_sopra") != -1): # (file.find("Taro") != -1 or file.find("Parma") != -1):  #
        print(file)
        file_inr1 = os.path.join(data_path, file )
        file_inr2 = os.path.join(file_inr1[:len(data_path)], "B8" + file_inr1[len(data_path)+2:])
        file_inr3= os.path.join(file_inr1[:len(data_path)] ,"B11" + file_inr1[len(data_path)+2:])
        file_inr4= os.path.join(file_inr1[:len(data_path)] ,"RF" + file_inr1[len(data_path)+2:])
        dataset = gdal.Open(file_inr1, gdal.GA_ReadOnly)
        b4 = dataset.ReadAsArray()
        # gvv_0 = -10*np.log10(gvv_0)
        dataset = None
        dataset = gdal.Open(file_inr2, gdal.GA_ReadOnly)
        b8 = dataset.ReadAsArray()
        dataset = None
        dataset = gdal.Open(file_inr3, gdal.GA_ReadOnly)
        b11 = dataset.ReadAsArray()
        dataset = None
        dataset = gdal.Open(file_inr4, gdal.GA_ReadOnly)
        rf = dataset.ReadAsArray()
        dataset = None
        veg = rf == 0
        sed = rf == 1
        wat = rf == 2
        mask_test = np.zeros((b8.shape[0], b8.shape[1]))
        mask_test[700:700 + 256, 70:70 + 128] = 1
        park_mask = mask_test 


#    #### other additional input   
        [s1, s2] = b8.shape
        print(s1)
        print(s2)
        p2 = []
        for y in range(1,s1-ps+1,r): 
            for x in range(1,s2-ps+1,r):
                mask_d0 = park_mask[y:y+ps,x:x+ps]
                [m1,m2] = mask_d0.shape
                s_0 =  mask_d0.sum()
                if s_0 == 0 : #s_0*s_1 == 0 : 
                    p2.append([y,x])
        p_train = []
        p_val = []
        print(len(p2))
        p = p2#[p2[s] for s in v]  
        random.shuffle(p)
        P = len(p2)
        p_train, p_val= p[:int(0.95*P)],p[int(0.95*P):P]
        print(len(p_train))
        print(len(p_val))
        
        y_train_k = np.ndarray(shape=( ps, ps, Out), dtype='float32')

        x_gtrain_k = np.ndarray(shape=( ps, ps, N), dtype='float32')

        
        # n = 0
        for patch in p_train:
            y0, x0 = patch[0], patch[1]
            x_gtrain_k[ :,:,0] = b8[y0:y0+ps,x0:x0+ps]/2000
            x_gtrain_k[ :,:,1] = b4[y0:y0+ps,x0:x0+ps]/2000
            x_gtrain_k[ :,:,2] = b11[y0:y0+ps,x0:x0+ps]/2000 

            y_train_k[ :,:,0]= veg[y0:y0+ps, x0:x0+ps]  # np.invert()#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_train_k[ :,:,1] = sed[y0:y0+ps, x0:x0+ps] #  np.invert(ndvi[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_train_k[ :,:,2] = wat[y0:y0+ps, x0:x0+ps]  # np.invert()#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_train_k[ :,:,3] = b4[y0:y0+ps, x0:x0+ps]/2000 #  np.invert(ndvi[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_train_k[ :,:,4] = b11[y0:y0+ps, x0:x0+ps]/2000  # np.invert()#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            np.save(os.path.join(folder_out_2,'X_train_' + str(final_out) + '.npy'),x_gtrain_k)
            np.save(os.path.join(folder_out_2,'Y_train_' + str(final_out) + '.npy'),y_train_k)
            final_out += 1
        
        
        x_val_k = np.ndarray(shape=( ps, ps, N), dtype='float32')
        y_val_k = np.ndarray(shape=( ps, ps, Out), dtype='float32')

        x_gval_k = np.ndarray(shape=( ps, ps, N), dtype='float32')

        x_bval_k = np.ndarray(shape=( ps, ps, N), dtype='float32')


        for patch in p_val:
            y0, x0 = patch[0], patch[1]

            x_gval_k[ :,:,0] = b8[y0:y0+ps,x0:x0+ps]/2000 
            x_gval_k[ :,:,1] = b4[y0:y0+ps,x0:x0+ps]/2000 
            x_gval_k[ :,:,2] = b11[y0:y0+ps,x0:x0+ps]/2000 

            
            y_val_k[ :,:,0]= veg[y0:y0+ps, x0:x0+ps]  # np.invert(cndvi[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_val_k[ :,:,1] = sed[y0:y0+ps, x0:x0+ps] #  np.invert(ndvi[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_val_k[ :,:,2] = wat[y0:y0+ps, x0:x0+ps]  # np.invert(mndwi[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_val_k[ :,:,3] = b4[y0:y0+ps, x0:x0+ps]/2000 #  np.invert(ndvi[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_val_k[ :,:,4] = b11[y0:y0+ps, x0:x0+ps]/2000  # np.invert(mndwi[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            np.save(os.path.join(folder_out_2,'X_val_' + str(final_out_2) + '.npy'),x_gval_k)
            np.save(os.path.join(folder_out_2,'Y_val_' + str(final_out_2) + '.npy'),y_val_k)
            final_out_2 += 1
#         #     n = n + 1
#         x_test[ n, :,:,0] = b8[700:700 + 256, 70:70 + 128]/2000 
#         x_test[n, :,:,1] = b4[700:700 + 256, 70:70 + 128]/2000 
#         x_test[ n,:,:,2] = b11[700:700 + 256, 70:70 + 128]/2000 

        
#         y_test[ n,:,:,0]= veg[700:700 + 256, 70:70 + 128]/2000 # np.invert(cndvi[700:700 + 128, 70:70 + 128])
#         y_test[ n,:,:,1] = sed[700:700 + 256, 70:70 + 128]/2000  # np.invert(ndvi[700:700 + 128, 70:70 + 128])
#         y_test[ n,:,:,2] = wat[700:700 + 256, 70:70 + 128]/2000 # np.invert( mndwi[700:700 + 128, 70:70 + 128])
#         y_test[ n,:,:,3] = b4[700:700 + 256, 70:70 + 128]/2000  # np.invert(ndvi[700:700 + 128, 70:70 + 128])
#         y_test[ n,:,:,4] = b11[700:700 + 256, 70:70 + 128]/2000 # np.invert( mndwi[700:700 + 128, 70:70 + 128])
#         n += 1


# np.savez(os.path.join(folder_out_2,"test_intradate_beta.npz"),x_test = x_test,y_test = y_test) #, x_gtrain = x_gtrain, y_train = y_train,  x_gval = x_gval, y_val = y_val)    
# n = 0 
for file in dir_list: 
   if file.find("B4") != -1 and (file.find("Enza") != -1) : # (file.find("Taro") != -1 or file.find("Parma") != -1):  #(file.find("Enza") != -1 or file.find("canneto_sopra") != -1): 
        print(file)
        file_inr1 = os.path.join(data_path, file )
        file_inr2 = os.path.join(file_inr1[:len(data_path)], "B8" + file_inr1[len(data_path)+2:])
        file_inr3= os.path.join(file_inr1[:len(data_path)] ,"B11" + file_inr1[len(data_path)+2:])
        file_inr4= os.path.join(file_inr1[:len(data_path)] ,"RF" + file_inr1[len(data_path)+2:])
        dataset = gdal.Open(file_inr1, gdal.GA_ReadOnly)
        b4 = dataset.ReadAsArray()
        # gvv_0 = -10*np.log10(gvv_0)
        dataset = None
        dataset = gdal.Open(file_inr2, gdal.GA_ReadOnly)
        b8 = dataset.ReadAsArray()
        dataset = None
        dataset = gdal.Open(file_inr3, gdal.GA_ReadOnly)
        b11 = dataset.ReadAsArray()
        dataset = None
        dataset = gdal.Open(file_inr4, gdal.GA_ReadOnly)
        rf = dataset.ReadAsArray()
        dataset = None
        veg = rf == 0
        sed = rf == 1
        wat = rf == 2
        mask_test = np.zeros((b8.shape[0], b8.shape[1]))
        mask_test[700:700 + 256, 70:70 + 128] = 1
        park_mask = mask_test 

        y_test = np.ndarray(shape=(1, b8.shape[0],b8.shape[1], Out), dtype='float32')
        x_test = np.ndarray(shape=(1, b8.shape[0],b8.shape[1], N), dtype='float32')

#    #### other additional input   
        #     n = n + 1
        x_test[ n, :,:,0] = b8[:,:]/2000 
        x_test[n, :,:,1] = b4[:,:]/2000 
        x_test[ n,:,:,2] = b11[:,:]/2000 

        
        y_test[ n,:,:,0]= veg[:,:] # np.invert(cndvi[700:700 + 128, 70:70 + 128])
        y_test[ n,:,:,1] = sed[:,:]  # np.invert(ndvi[700:700 + 128, 70:70 + 128])
        y_test[ n,:,:,2] = wat[:,:]# np.invert( mndwi[700:700 + 128, 70:70 + 128])
        y_test[ n,:,:,3] = b4[:,:] # np.invert(ndvi[700:700 + 128, 70:70 + 128])
        y_test[ n,:,:,4] = b11[:,:]/2000 # np.invert( mndwi[700:700 + 128, 70:70 + 128])
        n1 += 1


        np.savez(os.path.join(folder_out_2,"test_intradate_beta"+ str(n1) + ".npz"),x_test = x_test,y_test = y_test) #, x_gtrain = x_gtrain, y_train = y_train,  x_gval = x_gval, y_val = y_val)    

