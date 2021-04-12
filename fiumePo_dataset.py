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

path = r"C:\Users\massi\Downloads\drive-download-20200416T155818Z-001\\" #r"C:\Users\massi\Downloads\test1\\" #   D:\Albufera_2019_processed\subset_albufera\Dataset\\"
folder_out_2 = r"C:\Users\massi\Downloads\sentinel1 Salerno\train_data_\\" # r"C:\Users\massi\Downloads\sentinel1 Salerno\test_data1\\"#  

if not os.path.exists(folder_out_2):
    os.makedirs(folder_out_2)

#############
dir_list = os.listdir(path)
dir_list.sort()
print(dir_list)
ps = 128 

N = 2
Out = 2
import random
# x_train = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
# x_btrain = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
# x_gtrain = np.ndarray(shape=(0, ps, ps, N), dtype='float32')

# y_train = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')

# x_val = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
# x_bval = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
# x_gval = np.ndarray(shape=(0, ps, ps, N), dtype='float32')

# y_val = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')

files_gVH = []
files_gVV = []


files_MNDWIp = []

final_out = 0
final_out_2 = 0 

for file in dir_list: 
    # if file.find("sVV") != -1: 
    #     files_VV.append(file)
    # if file.find("sVH") != -1: 
    #     files_VH.append(file)
    print(file)
    if file.find("VV") != -1: 
        files_gVV.append(file)
    if file.find("VH") != -1: 
        files_gVH.append(file)
        
    # if file.find("VV") != -1: 
    #     files_bVV.append(file)
    # if file.find("VH") != -1: 
    #     files_bVH.append(file)
        
    # if file.find("NDVI_2") != -1: 
    #     files_NDVI.append(file)
    # if file.find("MNDWI_2") != -1: 
    #     files_MNDWI.append(file)
    if file.find("NDWI_Th") != -1: 
        files_MNDWIp.append(file)

print(len(files_gVV))
print(len(files_gVH))
print(len(files_MNDWIp))

# park_file = "park2.tif"
# dataset = gdal.Open(path + park_file, gdal.GA_ReadOnly)
# park2 = dataset.ReadAsArray()
# dataset = None
# park2 = park2/255
# park = 1 - park2

# [s1,s2] = park.shape
def open_image(image_path):

    image = gdal.Open(image_path)
    band = image.GetRasterBand(1)
    data_type = band.DataType
    cols = image.RasterXSize
    rows = image.RasterYSize
    # print(str(cols) + " " + str(rows))
    geotransform = image.GetGeoTransform()
    proj = image.GetProjection()
    # print(geotransform)
    # print(proj)
    minx = geotransform[0]
    maxy = geotransform[3]
    # print(str(minx) + " " + str(maxy))
    maxx = minx + geotransform[1] * cols
    miny = maxy + geotransform[5] * rows
    # print(str(maxx) + " " + str(miny))
    X_Y_raster_size = [cols, rows]
    # print(X_Y_raster_size)
    extent = [minx, miny, maxx, maxy]
    # print(extent)
    information = {}
    information['geotransform'] = geotransform
    information['extent'] = extent
    information['X_Y_raster_size'] = X_Y_raster_size
    information['projection'] = proj
    information['data_type'] = data_type
    image_array = np.array(image.ReadAsArray(0, 0, cols, rows))
    return image_array, information

y_test = np.ndarray(shape=(len(files_MNDWIp), 128, 128, Out), dtype='float32')
x_test = np.ndarray(shape=(len(files_MNDWIp), 128, 128, N), dtype='float32')

# print(y_test.shape)
veg = 0 
wat = 0 
noveg = 0
# r = 32
n = 0

# mndwi2 = np.zeros((park2.shape[0], park2.shape[1]))
# train = np.zeros((park2.shape[0], park2.shape[1]))
# val = np.zeros((park2.shape[0], park2.shape[1]))
for MNDWI_ind in range(len(files_MNDWIp)): 
    file_MNDWIp = os.path.join(path, files_MNDWIp[MNDWI_ind])

    dataset = gdal.Open(file_MNDWIp, gdal.GA_ReadOnly)
    mndwip = dataset.ReadAsArray()
    dataset = None
    a, b = open_image(file_MNDWIp)
    file_VV_0 = os.path.join(path, files_gVV[MNDWI_ind])
    file_VH_0 = os.path.join(path, files_gVH[MNDWI_ind])
    dataset = gdal.Open(file_VV_0, gdal.GA_ReadOnly)
    gvv_0 = dataset.ReadAsArray()
    # gvv_0 = -10*np.log10(gvv_0)
    dataset = None
    dataset = gdal.Open(file_VH_0, gdal.GA_ReadOnly)
    gvh_0 = dataset.ReadAsArray()
    # gvh_0 = -10*np.log10(gvh_0)
    dataset = None
    r = 32


#    #### other additional input   
    [s1, s2] = gvv_0.shape
    # print(gvv_0.shape)
    p2 = []
    for y in range(0,s1-ps+1,r): 
        for x in range(0,s2-ps+1,r):
            p2.append([y,x])
    p_train = []
    p_val = []
    P = len(p2)
    p = p2#[p2[s] for s in v]  
    # random.shuffle(p)
    p_train, p_val= p,p # p[:int(0.95*P)],p[int(0.95*P):P]
    
    y_train_k = np.ndarray(shape=( ps, ps, Out), dtype='float32')

    x_gtrain_k = np.ndarray(shape=( ps, ps, N), dtype='float32')

    
    # n = 0
    for patch in p_train:
        y0, x0 = patch[0], patch[1]
        x_gtrain_k[ :,:,0] = gvv_0[y0:y0+ps,x0:x0+ps]
        x_gtrain_k[ :,:,1] = gvh_0[y0:y0+ps,x0:x0+ps]

        y_train_k[ :,:,0] = mndwip[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
        y_train_k[ :,:,1] = 1 - mndwip[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
        np.save(os.path.join(folder_out_2,'X_train_' + str(final_out) + '.npy'),x_gtrain_k)
        np.save(os.path.join(folder_out_2,'Y_train_' + str(final_out) + '.npy'),y_train_k)
        final_out += 1
    
    x_val_k = np.ndarray(shape=( ps, ps, N), dtype='float32')
    y_val_k = np.ndarray(shape=( ps, ps, Out), dtype='float32')

    x_gval_k = np.ndarray(shape=( ps, ps, N), dtype='float32')

    x_bval_k = np.ndarray(shape=( ps, ps, N), dtype='float32')


    for patch in p_val:
        y0, x0 = patch[0], patch[1]
        x_gval_k[ :,:,0] = gvv_0[y0:y0+ps,x0:x0+ps]
        x_gval_k[ :,:,1] = gvh_0[y0:y0+ps,x0:x0+ps]
        
        y_val_k[ :,:,0] = mndwip[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
        y_val_k[ :,:,1] = 1 - mndwip[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
        np.save(os.path.join(folder_out_2,'X_val_' + str(final_out_2) + '.npy'),x_gval_k)
        np.save(os.path.join(folder_out_2,'Y_val_' + str(final_out_2) + '.npy'),y_val_k)
        final_out_2 += 1
    #     n = n + 1
    # # x_test[ n, :,:,0] = gvv_0[1230:1230 + 128, 1080:1080 + 128]
    # # x_test[ n,:,:,1] = gvh_0[1230:1230 + 128, 1080:1080 + 128]

    
    # # y_test[ n,:,:,0] = mndwip[1230:1230 + 128, 1080:1080 + 128]
    # n += 1
    
    

# np.savez("test_fiumePo.npz",x_test = x_test,y_test = y_test) #, x_gtrain = x_gtrain, y_train = y_train,  x_gval = x_gval, y_val = y_val)    