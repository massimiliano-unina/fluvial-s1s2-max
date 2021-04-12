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

data_path = r"D:\fiumiunsupervised\drive-download-20210125T123716Z-001\out\\" #"D:\fiumi unsupervised\0_Borgoforte1\\" #D:\Albufera_2019_processed\subset_albufera\Dataset\\"
data_path2 = r"D:\fiumiunsupervised\drive-download-20210125T123716Z-001\out\\" # #D:\Albufera_2019_processed\subset_albufera\Dataset\\"

# folder_out_2 = r"D:\fiumiunsupervised\Training_Set_Unsupervised_Borgoforte3\\"

folder_out_2 = r"D:\fiumiunsupervised\Training_Set_Unsupervised_Borgoforte_withOptNIR\\"

if not os.path.exists(folder_out_2):
    os.makedirs(folder_out_2)

#############
dir_list = os.listdir(data_path)
dir_list.sort()
print(dir_list)

dir_list2 = os.listdir(data_path2)
dir_list2.sort()
print(dir_list2)


ps = 128 

N = 6 #2
Out = 3 + 4
import random

# print(y_test.shape)
veg = 0 
wat = 0 
noveg = 0
r = 32
n = 0
final_out = 0
final_out_2 = 0
count_veg = 0
count_sed = 0
count_wat = 0 
## CONTEGGIO PIXEL CLASSE PER CLASSE
# for file in dir_list: 
#    if file.find("VV_Po_S1_pre_") != -1:
#         print(file)
#         name_ = 13
#         file_inr1_pre = os.path.join(data_path, file )
#         # file_inr2 = os.path.join(file_inr1[:len(data_path)], "B8" + file_inr1[len(data_path)+6:])
#         file_inr4= os.path.join(file_inr1_pre[:len(data_path)] ,"RF_Po_S1S2_" + file_inr1_pre[len(data_path)+name_:])

#         dataset = gdal.Open(file_inr4, gdal.GA_ReadOnly)
#         rf = dataset.ReadAsArray()
#         dataset = None
#         veg = rf == 0
#         sed = rf == 1
#         wat = rf == 2
#         count_veg += np.sum(veg)
#         count_sed += np.sum(sed)
#         count_wat += np.sum(wat)
# print(count_veg/(count_sed + count_veg + count_wat))
# print(count_sed/(count_sed + count_veg + count_wat))
# print(count_wat/(count_sed + count_veg + count_wat))


## CREAZIONE DATASET 

for file in dir_list: 
   if file.find("VV_Po_S1_pre_") != -1:
        print(file)
        file_inr1_pre = os.path.join(data_path, file )
        name_ = 13

        # file_inr2 = os.path.join(file_inr1[:len(data_path)], "B8" + file_inr1[len(data_path)+6:])
        file_inr4= os.path.join(file_inr1_pre[:len(data_path)] ,"RF_Po_S1S2_" + file_inr1_pre[len(data_path)+name_:])
        file_inr5= os.path.join(file_inr1_pre[:len(data_path)] ,"VH_Po_S1_" + file_inr1_pre[len(data_path)+name_:])
        file_inr5_pre= os.path.join(file_inr1_pre[:len(data_path)] ,"VH_Po_S1_pre_" + file_inr1_pre[len(data_path)+name_:])
        file_inr5_post= os.path.join(file_inr1_pre[:len(data_path)] ,"VH_Po_S1_post_" + file_inr1_pre[len(data_path)+name_:])
        file_inr1= os.path.join(file_inr1_pre[:len(data_path)] ,"VV_Po_S1_" + file_inr1_pre[len(data_path)+name_:])
        file_inr1_post= os.path.join(file_inr1_pre[:len(data_path)] ,"VV_Po_S1_post_" + file_inr1_pre[len(data_path)+name_:])
        # file_inr1_opt= os.path.join(file_inr1_pre[:len(data_path)] ,"RGB_Po_S2_" + file_inr1_pre[len(data_path)+name_:])
        file_inr1_opt= os.path.join(file_inr1_pre[:len(data_path)] ,"RGBN_Po_S2_" + file_inr1_pre[len(data_path)+name_:])
# "D:\fiumiunsupervised\drive-download-20210125T123716Z-001\out\RGB_Osti_S2_2018-09-16.tif"
# "D:\fiumiunsupervised\drive-download-20210125T123716Z-001\out\RGB_Po_S2_2018-09-16.tif"
        dataset = gdal.Open(file_inr1_opt, gdal.GA_ReadOnly)
        rgb = dataset.ReadAsArray()
        # gvv_0 = -10*np.log10(gvv_0)
        dataset = None
        dataset = gdal.Open(file_inr1, gdal.GA_ReadOnly)
        vv = dataset.ReadAsArray()
        # gvv_0 = -10*np.log10(gvv_0)
        dataset = None
        dataset = gdal.Open(file_inr5, gdal.GA_ReadOnly)
        vh = dataset.ReadAsArray()
        # gvv_0 = -10*np.log10(gvv_0)
        dataset = None
        dataset = gdal.Open(file_inr1_pre, gdal.GA_ReadOnly)
        vv0 = dataset.ReadAsArray()
        # gvv_0 = -10*np.log10(gvv_0)
        dataset = None
        dataset = gdal.Open(file_inr1_post, gdal.GA_ReadOnly)
        vv2 = dataset.ReadAsArray()
        # gvv_0 = -10*np.log10(gvv_0)
        dataset = None
        
        dataset = gdal.Open(file_inr5_post, gdal.GA_ReadOnly)
        vh2 = dataset.ReadAsArray()
        # gvv_0 = -10*np.log10(gvv_0)
        dataset = None
        dataset = gdal.Open(file_inr5_pre, gdal.GA_ReadOnly)
        vh0 = dataset.ReadAsArray()
        # gvv_0 = -10*np.log10(gvv_0)
        dataset = None

        dataset = gdal.Open(file_inr4, gdal.GA_ReadOnly)
        rf = dataset.ReadAsArray()
        dataset = None
        print(np.unique(rf))
        veg = rf == 0
        sed = rf == 1
        wat = rf == 2
        mask_test = np.zeros((vv.shape[0], vv.shape[1]))
        # mask_test[700:700 + 256, 70:70 + 128] = 1
        # park_mask = mask_test 


#    #### other additional input   
        [s1, s2] = vv.shape
        # print(s1)
        # print(s2)
        p2 = []
        for y in range(1,s1-ps+1,r): 
            for x in range(1,s2-ps+1,r):
                mask_d0 = mask_test[y:y+ps,x:x+ps]
                [m1,m2] = mask_d0.shape
                s_0 =  mask_d0.sum()
                if s_0 == 0 : #s_0*s_1 == 0 : 
                    p2.append([y,x])
        p_train = []
        p_val = []
        # print(len(p2))
        p = p2#[p2[s] for s in v]  
        random.shuffle(p)
        P = len(p2)
        p_train, p_val= p[:int(0.95*P)],p[int(0.95*P):P]
        print(len(p_train))
        print(len(p_val))
        
        y_train_k = np.ndarray(shape=( ps, ps, Out), dtype='float32')

        x_gtrain_k = np.ndarray(shape=( ps, ps, N), dtype='float32')
        # print(vv.shape)
        # print(vv0.shape)
        # print(vv2.shape)
        print(rgb.shape)
        
        # n = 0
        for patch in p_train:
            y0, x0 = patch[0], patch[1]
            x_gtrain_k[ :,:,0] = vv[y0:y0+ps,x0:x0+ps]
            x_gtrain_k[ :,:,1] = vh[y0:y0+ps,x0:x0+ps]
            x_gtrain_k[ :,:,2] = vv0[y0:y0+ps,x0:x0+ps]
            x_gtrain_k[ :,:,3] = vh0[y0:y0+ps,x0:x0+ps]
            x_gtrain_k[ :,:,4] = vv2[y0:y0+ps,x0:x0+ps]
            x_gtrain_k[ :,:,5] = vh2[y0:y0+ps,x0:x0+ps]
            
            y_train_k[ :,:,0]= veg[y0:y0+ps, x0:x0+ps]  # np.invert()#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_train_k[ :,:,1] = sed[y0:y0+ps, x0:x0+ps] #  np.invert(ndvi[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_train_k[ :,:,2] = wat[y0:y0+ps, x0:x0+ps] # np.invert()#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_train_k[ :,:,3]= np.squeeze(rgb[0,y0:y0+ps, x0:x0+ps]) # np.invert()#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_train_k[ :,:,4] = np.squeeze(rgb[1,y0:y0+ps, x0:x0+ps]) #  np.invert(ndvi[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_train_k[ :,:,5] = np.squeeze(rgb[2, y0:y0+ps, x0:x0+ps]) # np.invert()#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_train_k[ :,:,6] = np.squeeze(rgb[3, y0:y0+ps, x0:x0+ps]) # np.invert()#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]

            np.save(os.path.join(folder_out_2,'X_train_' + str(final_out) + '.npy'),x_gtrain_k)
            np.save(os.path.join(folder_out_2,'Y_train_' + str(final_out) + '.npy'),y_train_k)
            final_out += 1
        
        
        y_val_k = np.ndarray(shape=( ps, ps, Out), dtype='float32')

        x_gval_k = np.ndarray(shape=( ps, ps, N), dtype='float32')



        for patch in p_val:
            y0, x0 = patch[0], patch[1]

            x_gval_k[ :,:,0] = vv[y0:y0+ps,x0:x0+ps]
            x_gval_k[ :,:,1] = vh[y0:y0+ps,x0:x0+ps]
            x_gval_k[ :,:,2] = vv0[y0:y0+ps,x0:x0+ps]
            x_gval_k[ :,:,3] = vh0[y0:y0+ps,x0:x0+ps]
            x_gval_k[ :,:,4] = vv2[y0:y0+ps,x0:x0+ps]
            x_gval_k[ :,:,5] = vh2[y0:y0+ps,x0:x0+ps]


            y_val_k[ :,:,0]= veg[y0:y0+ps, x0:x0+ps]  # np.invert(cndvi[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_val_k[ :,:,1] = sed[y0:y0+ps, x0:x0+ps]#  np.invert(ndvi[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_val_k[ :,:,2] = wat[y0:y0+ps, x0:x0+ps]  # np.invert(mndwi[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_val_k[ :,:,3]= np.squeeze(rgb[0,y0:y0+ps, x0:x0+ps])  # np.invert(cndvi[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_val_k[ :,:,4] = np.squeeze(rgb[1,y0:y0+ps, x0:x0+ps])#  np.invert(ndvi[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_val_k[ :,:,5] = np.squeeze(rgb[2,y0:y0+ps, x0:x0+ps])  # np.invert(mndwi[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_val_k[ :,:,6] = np.squeeze(rgb[3,y0:y0+ps, x0:x0+ps])  # np.invert(mndwi[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            np.save(os.path.join(folder_out_2,'X_val_' + str(final_out_2) + '.npy'),x_gval_k)
            np.save(os.path.join(folder_out_2,'Y_val_' + str(final_out_2) + '.npy'),y_val_k)
            final_out_2 += 1

n1 = 0
for file in dir_list2: 
   if file.find("VV_Osti_S1_pre_") != -1:
        print(file)
        file_inr1_pre = os.path.join(data_path, file )
        name_ = 15
        # file_inr2 = os.path.join(file_inr1[:len(data_path)], "B8" + file_inr1[len(data_path)+6:])
        file_inr4= os.path.join(file_inr1_pre[:len(data_path)] ,"RF_Osti_S1S2_" + file_inr1_pre[len(data_path)+name_:])
        file_inr5= os.path.join(file_inr1_pre[:len(data_path)] ,"VH_Osti_S1_" + file_inr1_pre[len(data_path)+name_:])
        file_inr5_pre= os.path.join(file_inr1_pre[:len(data_path)] ,"VH_Osti_S1_pre_" + file_inr1_pre[len(data_path)+name_:])
        file_inr5_post= os.path.join(file_inr1_pre[:len(data_path)] ,"VH_Osti_S1_post_" + file_inr1_pre[len(data_path)+name_:])
        file_inr1= os.path.join(file_inr1_pre[:len(data_path)] ,"VV_Osti_S1_" + file_inr1_pre[len(data_path)+name_:])
        file_inr1_post= os.path.join(file_inr1_pre[:len(data_path)] ,"VV_Osti_S1_post_" + file_inr1_pre[len(data_path)+name_:])
        # file_inr1_opt= os.path.join(file_inr1_pre[:len(data_path)] ,"RGB_Osti_S2_" + file_inr1_pre[len(data_path)+name_:])
        file_inr1_opt= os.path.join(file_inr1_pre[:len(data_path)] ,"RGBN_Osti_S2_" + file_inr1_pre[len(data_path)+name_:])
        dataset = gdal.Open(file_inr1, gdal.GA_ReadOnly)
        vv = dataset.ReadAsArray()
        # gvv_0 = -10*np.log10(gvv_0)
        dataset = None
        dataset = gdal.Open(file_inr1_opt, gdal.GA_ReadOnly)
        rgb = dataset.ReadAsArray()
        # gvv_0 = -10*np.log10(gvv_0)
        dataset = None
        
        dataset = gdal.Open(file_inr5, gdal.GA_ReadOnly)
        vh = dataset.ReadAsArray()
        # gvv_0 = -10*np.log10(gvv_0)
        dataset = None
        dataset = gdal.Open(file_inr1_pre, gdal.GA_ReadOnly)
        vv0 = dataset.ReadAsArray()
        # gvv_0 = -10*np.log10(gvv_0)
        dataset = None
        dataset = gdal.Open(file_inr1_post, gdal.GA_ReadOnly)
        vv2 = dataset.ReadAsArray()
        # gvv_0 = -10*np.log10(gvv_0)
        dataset = None
        
        dataset = gdal.Open(file_inr5_post, gdal.GA_ReadOnly)
        vh2 = dataset.ReadAsArray()
        # gvv_0 = -10*np.log10(gvv_0)
        dataset = None
        dataset = gdal.Open(file_inr5_pre, gdal.GA_ReadOnly)
        vh0 = dataset.ReadAsArray()
        # gvv_0 = -10*np.log10(gvv_0)
        dataset = None
        dataset = gdal.Open(file_inr4, gdal.GA_ReadOnly)
        rf = dataset.ReadAsArray()
        dataset = None
        veg = rf == 0
        sed = rf == 1
        wat = rf == 2
        mask_test = np.zeros((vv.shape[0], vv.shape[1]))
        # mask_test[700:700 + 256, 70:70 + 128] = 1
        # park_mask = mask_test 

        y_test = np.ndarray(shape=(1, vv.shape[0],vv.shape[1], Out), dtype='float32')
        x_test = np.ndarray(shape=(1, vv.shape[0],vv.shape[1], N), dtype='float32')

#    #### other additional input   
        #     n = n + 1
        x_test[ n, :,:,0] = vv[:,:]
        x_test[ n, :,:,1] = vh[:,:]
        x_test[ n, :,:,2] = vv0[:,:]
        x_test[ n, :,:,3] = vh0[:,:]
        x_test[ n, :,:,4] = vv2[:,:]
        x_test[ n, :,:,5] = vh2[:,:]

        y_test[ n,:,:,0]= veg[:,:] # np.invert(cndvi[700:700 + 128, 70:70 + 128])
        y_test[ n,:,:,1] = sed[:,:]  # np.invert(ndvi[700:700 + 128, 70:70 + 128])
        y_test[ n,:,:,2] = wat[:,:] # np.invert( mndwi[700:700 + 128, 70:70 + 128])
        y_test[ n,:,:,3]= np.squeeze(rgb[0,:,:]) # np.invert(cndvi[700:700 + 128, 70:70 + 128])
        y_test[ n,:,:,4] = np.squeeze(rgb[1,:,:]) # np.invert(ndvi[700:700 + 128, 70:70 + 128])
        y_test[ n,:,:,5] = np.squeeze(rgb[2,:,:]) # np.invert( mndwi[700:700 + 128, 70:70 + 128])
        y_test[ n,:,:,6] = np.squeeze(rgb[3,:,:]) # np.invert( mndwi[700:700 + 128, 70:70 + 128])

        n1 += 1


        np.savez(os.path.join(folder_out_2,"test_intradate1_beta"+ str(n1) + ".npz"),x_test = x_test,y_test = y_test) #, x_gtrain = x_gtrain, y_train = y_train,  x_gval = x_gval, y_val = y_val)    

