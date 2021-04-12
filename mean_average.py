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
import scipy.signal
from scipy.ndimage import gaussian_filter, morphology,generate_binary_structure
from scipy.io import savemat
import gdal
from matplotlib import pyplot as plt
    #############
    # short names

path = r"D:\Albufera_2019_processed\Dataset_paper_2020\\"
# folder_out_2 = r"D:\Albufera_2019_processed\Sigma2\\"
# folder_out_2 = r"D:\Albufera_2019_processed\Gamma2\\"
folder_out_2 = r"D:\Albufera_2019_processed\Beta2\\"

if not os.path.exists(folder_out_2):
    os.makedirs(folder_out_2)

#############
dir_list = os.listdir(path)
dir_list.sort()
print(dir_list)

import time 

ps = 128 

N = 6
Out = 5
import random

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

final_out = 0
final_out_2 = 0 

for file in dir_list: 
    if file.find("201911") == -1 and file.find("201912") == -1 :  ############# 
    # if file.find("sVV") != -1: 
    #     files_VV.append(file)
    # if file.find("sVH") != -1: 
    #     files_VH.append(file)
        if file.find("_VV") != -1: 
            files_gVV.append(file)
        if file.find("_VH") != -1: 
            files_gVH.append(file)
        # if file.find("bVV") != -1: 
        #     files_bVV.append(file)
        # if file.find("bVH") != -1: 
        #     files_bVH.append(file)
            
        # if file.find("NDVI_2") != -1: 
        #     files_NDVI.append(file)
        # if file.find("MNDWI_2") != -1: 
        #     files_MNDWI.append(file)
        if file.find("NDVI_pre") != -1: 
            files_NDVIp.append(file)
        if file.find("MNDWI_pre") != -1: 
            files_MNDWIp.append(file)

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

lake_file = "Lake.tif"
dataset = gdal.Open(path + lake_file, gdal.GA_ReadOnly)
lake = dataset.ReadAsArray()
dataset = None

park2 = park2/255
# print(np.max(park2))
park = 1 - park2

[s1,s2] = park.shape


veg = 0 
wat = 0 
noveg = 0
r = 32
n = 0

vvm_ts = []
vhm_ts = []
vvn_ts = []
vhn_ts = []

dvvm_ts = []
dvhm_ts = []
dvvn_ts = []
dvhn_ts = []

vvm_ts2 = []
vhm_ts2 = []
vvn_ts2 = []
vhn_ts2 = []

dvvm_ts2 = []
dvhm_ts2 = []
dvvn_ts2 = []
dvhn_ts2 = []

vvm_ts3 = []
vhm_ts3 = []

c = 0
VV_V = []
VH_V = []
VV_M = []
VH_M = []
VV_N = []
VH_N = []
VV_C =[]
VH_C = []
 
for MNDWI_ind in range(len(files_MNDWIp)): 
    init = time.time()
    print(c)
    # file_MNDWI = os.path.join(path, files_MNDWI[MNDWI_ind])
    # file_NDVI = os.path.join(path, files_NDVI[MNDWI_ind])
    file_MNDWIp = os.path.join(path, files_MNDWIp[MNDWI_ind])
    file_NDVIp = os.path.join(path, files_NDVIp[MNDWI_ind])

    dataset = gdal.Open(file_MNDWIp, gdal.GA_ReadOnly)
    mndwip = dataset.ReadAsArray()
    mndwi = mndwip > 0
    # print(np.max(mndwi))
    mndwi2 =  np.invert(mndwi) # 1 - mndwi #
    dataset = None
    dataset = gdal.Open(file_NDVIp, gdal.GA_ReadOnly)
    ndvip = dataset.ReadAsArray()
    ndvi =  ndvip > 0.3
    ndvi2 = np.invert(ndvi) #  1 - ndvi #
    dataset = None
    # dataset = gdal.Open(file_MNDWIp, gdal.GA_ReadOnly)
    # mndwip = dataset.ReadAsArray()
    # dataset = None
    # dataset = gdal.Open(file_NDVIp, gdal.GA_ReadOnly)
    # ndvip = dataset.ReadAsArray()
    # dataset = None

    file_VV_0 = os.path.join(path, files_gVV[3*MNDWI_ind])
    file_VV_1 = os.path.join(path, files_gVV[3*MNDWI_ind + 1])
    file_VV_2 = os.path.join(path, files_gVV[3*MNDWI_ind + 2])
    file_VH_0 = os.path.join(path, files_gVH[3*MNDWI_ind])
    file_VH_1 = os.path.join(path, files_gVH[3*MNDWI_ind + 1])
    file_VH_2 = os.path.join(path, files_gVH[3*MNDWI_ind + 2])
    dataset = gdal.Open(file_VV_0, gdal.GA_ReadOnly)
    gvv_0 = dataset.ReadAsArray()
    # gvv_0 = gaussian_filter(gvv_0, (3,3))
    dataset = None
    dataset = gdal.Open(file_VV_1, gdal.GA_ReadOnly)
    gvv_1 = dataset.ReadAsArray()
    # gvv_1 = gaussian_filter(gvv_1, (3,3))
    dataset = None
    dataset = gdal.Open(file_VV_2, gdal.GA_ReadOnly)
    gvv_2 = dataset.ReadAsArray()
    # gvv_2 = gaussian_filter(gvv_2, (3,3))
    dataset = None
    dataset = gdal.Open(file_VH_0, gdal.GA_ReadOnly)
    gvh_0 = dataset.ReadAsArray()
    # gvh_0 = gaussian_filter(gvh_0, (3,3))
    dataset = None
    dataset = gdal.Open(file_VH_1, gdal.GA_ReadOnly)
    gvh_1 = dataset.ReadAsArray()
    # gvh_1 = gaussian_filter(gvh_1, (3,3))
    dataset = None
    dataset = gdal.Open(file_VH_2, gdal.GA_ReadOnly)
    gvh_2 = dataset.ReadAsArray()
    # gvh_2 = gaussian_filter(gvh_2, (3,3))
    dataset = None
    # print("aperte le immagini")
    cndvi = np.invert(ndvi)*np.invert(mndwi) #  # (1 - ndvi)*(1 - mndwi)
    cndvi2 = np.invert(cndvi)
    # print(mndwi.shape)
    # print(np.max(mndwi))
    print("VH " + file_VH_1)
    print("VV " + file_VV_1)
    kernel = np.ones((5,5))
    print(kernel.shape)
    kernel /= 25
    print("Max of VH " + str(np.mean(gvh_1)))
    gvh_11 = scipy.signal.convolve2d(gvh_1, kernel, mode='same') # gvh_1 # 
    print("Post Max of VH " + str(np.mean(gvh_11)))
    gvh_11 = -10*np.log10(gvh_11 + 10**(-16))
    print("Max of VV " + str(np.mean(gvv_1)))
    gvv_11 = scipy.signal.convolve2d(gvv_1, kernel, mode='same') # gvv_1 # 
    print("Post Max of VV " + str(np.mean(gvv_11)))
    gvv_11 = -10*np.log10(gvv_11 + 10**(-16))
    lake2 = lake > 0 
    # mask_gvh_1 = (gvh_11 > 21.55)*(gvv_11 > 14.65) + lake2
    mask_gvh_1 = (gvh_11 / gvv_11) < 1.42 
    mask_bare = (gvh_11 / gvv_11) >= 1.67 # np.invert(mask_gvh_1*mask_gvv_1) # 1 -  mask_gvh_1*mask_gvv_1 #
    # mask_gvh_1 = np.invert(mask_gvh_1) # 1 - mask_gvh_1 #
    # mask_gvh_1 = mask_gvh_1 > 0 
    # file_VH_1 = os.path.join(path, files_gVH[3*MNDWI_ind + 1])
    file_VH_11 =  os.path.join(path,str(c+1) +"Bare_Thre.tif")
    imsave(file_VH_11,mask_gvh_1)
    mask_gvv_1 = np.invert(mask_gvh_1*mask_bare) #((gvh_11 / gvv_11) >= 1.42)*((gvh_11 / gvv_11) < 1.67) #(gvh_11 < 17.47)*(gvv_11 > 10.43)*(gvv_11 <= 14.68)*(np.invert(lake2))
    # mask_gvv_1 =np.invert(mask_gvv_1) # 1 - mask_gvv_1 # np.invert(mask_gvv_1 > 0)
    # file_VV_1 = os.path.join(path, files_gVV[3*MNDWI_ind + 1])
    file_VH_11 =  os.path.join(path,str(c+1) +"Veg_Thre.tif")
    imsave(file_VH_11,mask_gvv_1)
    # file_VV_1 = os.path.join(path, files_gVV[3*MNDWI_ind + 1])
    file_V_11 =  os.path.join(path,str(c+1) + "Wat_Thre.tif")
    imsave(file_V_11,mask_bare)
    VV = gvv_1*park2*( 1 - lake)
    VH = gvh_1*park2*( 1 - lake)
    file_VV_11 =  os.path.join(path,str(c+1) + "VV_Thre.tif")
    imsave(file_VV_11,VV)
    file_VH_11 =  os.path.join(path,str(c+1) + "VH_Thre.tif")
    imsave(file_VH_11,VH)
    vv_mndwi = np.reshape(gvv_11*mndwi2*park2, newshape= (mndwi2.shape[0]*mndwi2.shape[1],1))
    vh_mndwi = np.reshape(gvh_11*mndwi2*park2, newshape= (mndwi2.shape[0]*mndwi2.shape[1],1))
    vv_ndvi = np.reshape(gvv_11*ndvi2*park2, newshape= (mndwi2.shape[0]*mndwi2.shape[1],1))
    vh_ndvi = np.reshape(gvh_11*ndvi2*park2, newshape= (mndwi2.shape[0]*mndwi2.shape[1],1))
    
    
    vvv = np.reshape(gvv_11*park2, newshape= (mndwi2.shape[0]*mndwi2.shape[1],1))
    vhv = np.reshape(gvh_11*park2, newshape= (mndwi2.shape[0]*mndwi2.shape[1],1))
    # print("mappe su cui calcolare statistiche ")
    # print( np.nonzero(vv_mndwi)[0])
    VV_M1 =  [vv_mndwi[kk] for kk in np.nonzero(vv_mndwi)[0]]
    VH_M1 =  [vh_mndwi[kk] for kk in np.nonzero(vh_mndwi)[0]]
    VV_N1 =  [vv_ndvi[kk] for kk in np.nonzero(vv_ndvi)[0]]
    VH_N1 =  [vh_ndvi[kk] for kk in np.nonzero(vh_ndvi)[0]]
    VV_V1 =  [vvv[kk] for kk in np.nonzero(vvv)[0]]
    VH_V1 =  [vhv[kk] for kk in np.nonzero(vhv)[0]]
    
    # print("valori diversi da zero")
    # VV_M = np.array(VV_M)
    # VH_M = np.array(VH_M)
    # VV_N = np.array(VV_N)
    # VH_N = np.array(VH_N)
    # VV_V1 = np.array(VV_V1)
    # VH_V1 = np.array(VH_V1)
    VV_V = VV_V + VV_V1
    VH_V = VH_V + VH_V1
    
    VV_M = VV_M + VV_M1
    VH_M = VH_M + VH_M1

    VV_N = VV_N + VV_N1
    VH_N = VH_N + VH_N1


    print(len(VV_V))
    # plt.figure()
    # plt.scatter(VV_M[::1000], VH_M[::1000])
    # # plt.hold(True)
    # plt.scatter(VV_N[::1000], VH_N[::1000])
    # plt.hold(True)
    print("vvm : " + str(np.mean(VV_M)))
    print("vhm : " + str(np.mean(VH_M)))
    print("vvn : " + str(np.mean(VV_N)))
    print("vhn : " + str(np.mean(VH_N)))
    
    print("MAX vvm : " + str(np.max(VV_M)))
    print("MAX vhm : " + str(np.max(VH_M)))
    print("MAX vvn : " + str(np.max(VV_N)))
    print("MAX vhn : " + str(np.max(VH_N)))
    
    print("min vvm : " + str(np.min(VV_M)))
    print("min vhm : " + str(np.min(VH_M)))
    print("min vvn : " + str(np.min(VV_N)))
    print("min vhn : " + str(np.min(VH_N)))
    vvm_ts.append(np.mean(VV_M))
    vhm_ts.append(np.mean(VH_M))
    vvn_ts.append(np.mean(VV_N))
    vhn_ts.append(np.mean(VH_N))
    vvm_ts3.append(np.mean(VV_V))
    vhm_ts3.append(np.mean(VH_V))
    
    # print("calcolata media")
    dvvm_ts.append(np.std(VV_M))
    dvhm_ts.append(np.std(VH_M))
    dvvn_ts.append(np.std(VV_N))
    dvhn_ts.append(np.std(VH_N))
    # print("calcolata deviazione standard")

    vv_cndwi = np.reshape(gvv_11*cndvi2*park2, newshape= (mndwi2.shape[0]*mndwi2.shape[1],1))
    vh_cndwi = np.reshape(gvh_11*cndvi2*park2, newshape= (mndwi2.shape[0]*mndwi2.shape[1],1))
    # print(vv_mndwi.shape)
    # print("mappe su cui calcolare statistiche ")
    # print( np.nonzero(vv_mndwi)[0])
    VV_C1 =  [vv_cndwi[kk] for kk in np.nonzero(vv_cndwi)[0]]
    VH_C1 =  [vh_cndwi[kk] for kk in np.nonzero(vh_cndwi)[0]]
    VV_C = VV_C + VV_C1 
    VH_C = VH_C + VH_C1
    # print("valori diversi da zero")
    # VV_C = np.array(VV_C)
    # VH_C = np.array(VH_C)
    print("vvn : " + str(np.mean(VV_N)))
    print("vhn : " + str(np.mean(VH_N)))

    print("MAX vvn : " + str(np.max(VV_N)))
    print("MAX vhn : " + str(np.max(VH_N)))

    print("min vvn : " + str(np.min(VV_N)))
    print("min vhn : " + str(np.min(VH_N)))

    # plt.scatter(VV_C[::1000], VH_C[::1000])
    vvm_ts2.append(np.mean(VV_C))
    vhm_ts2.append(np.mean(VH_C))
    # print("calcolata media")
    dvvm_ts2.append(np.std(VV_C))
    dvhm_ts2.append(np.std(VH_C))
    # print("calcolata deviazione standard")

    # vv_mndwi = np.reshape(gvv_11*park2*( 1 - lake), newshape= (mndwi.shape[0]*mndwi.shape[1],1))
    # vh_mndwi = np.reshape(gvh_11*park2*( 1 - lake), newshape= (mndwi.shape[0]*mndwi.shape[1],1))
    # # print(vv_mndwi.shape)
    # # print("mappe su cui calcolare statistiche ")
    # # print( np.nonzero(vv_mndwi)[0])
    # VV_M =  [vv_mndwi[kk] for kk in np.nonzero(vv_mndwi)[0]]
    # VH_M =  [vh_mndwi[kk] for kk in np.nonzero(vh_mndwi)[0]]
    # # print("valori diversi da zero")
    # VV_P = np.array(VV_M)
    # VH_P = np.array(VH_M)
    # # print(VV_M.shape)
    # vvm_ts3.append(np.mean(VV_P))
    # vhm_ts3.append(np.mean(VH_P))
    # print("calcolata media")
    c += 1
    final = time.time() - init
    print(final)
# from matplotlib.pyplot import plot as plt
dic = { "vvm_ts" : vvm_ts, "dvvm_ts" : dvvm_ts, "vhm_ts" : vhm_ts, "dvhm_ts" : dvhm_ts, "vvn_ts": vvn_ts, "dvvn_ts": dvvn_ts, "vhn_ts" : vhn_ts, "dvhn_ts" : dvhn_ts }
savemat('std_mean.mat', dic)

dic_no = {"VV_C": VV_C, "VH_C": VH_C,"VV_V": VV_V, "VH_V": VH_V, "VV_M": VV_M, "VH_M": VH_M, "VV_N": VV_N, "VH_N": VH_N}
savemat('behaviour.mat', dic_no)

dic2 = { "vvm_ts2" : vvm_ts2, "dvvm_ts2" : dvvm_ts2, "vhm_ts2" : vhm_ts2, "dvhm_ts2" : dvhm_ts2} #, "vvn_ts2": vvn_ts2, "dvvn_ts2": dvvn_ts2, "vhn_ts2" : vhn_ts2, "dvhn_ts2" : dvhn_ts2 }
savemat('std_mean2.mat', dic2)

dic3 = { "vvm_ts3" : vvm_ts3, "vhm_ts3" : vhm_ts3}
savemat('std_mean3.mat', dic3)
# plt.hold(False) 



plt.figure()
plt.plot(vhm_ts)
# plt.hold(True) 
plt.plot(vhn_ts)
plt.plot(vhm_ts2)
# plt.hold(False) 

plt.figure()
plt.plot(vvm_ts)
# plt.hold(True) 
plt.plot(vvn_ts)
plt.plot(vvm_ts2)
# plt.hold(False) 