# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:28:42 2019

@author: massi
"""
from main_functions import compute_fft 
import imageio 
from matplotlib import pyplot as plt
import numpy as np 
from tifffile import imsave 
import random


N = 2 
Out = 1
num = 1
r = 128 #32
ps = 128 #r


x_train = np.ndarray(shape=(0, ps,ps, N), dtype='float32')
y_train = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
x_val = np.ndarray(shape=(0, ps,ps, N), dtype='float32')
y_val = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')


#x_train2 = np.ndarray(shape=(0, ps,ps, N), dtype='float32')
#y_train2 = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
#x_val2 = np.ndarray(shape=(0, ps,ps, N), dtype='float32')
#y_val2 = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
#
#x_train3 = np.ndarray(shape=(0, ps,ps, N), dtype='float32')
#y_train3 = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
#x_val3 = np.ndarray(shape=(0, ps,ps, N), dtype='float32')
#y_val3 = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')


#a = [ "0_1_fatto", "1_1_fatto", "1_0_fatto2", "0_1_fatto2", "0_0_fatto2"]

a = []
for gg in range(4):
    for hh in range(4):
        a.append(str(gg) + "_" + str(hh))
#plt.figure()
c = 1
for n in a: 
    im3 = r"D:\Works\DLR\\brasil_GLAD_2017_2000_"+ n + ".tif"
    loss = imageio.imread(im3)
    loss = np.asarray(loss)
    
#    im = r"D:\Works\DLR\\brasil_XSAR_"+ n + ".tif"
#    xsar = imageio.imread(im)
#    xsar = np.asarray(xsar)

    im2 = r"D:\Works\DLR\\brasil_NASADEM_"+ n + ".tif"
    nasa = imageio.imread(im2)
    nasa = np.asarray(nasa)

#    im4 = r"D:\Works\DLR\\brasil_landsat_2015_"+ n + ".tif"
#    land15 = imageio.imread(im4)
#    land15 = np.asarray(land15)

    im = r"D:\Works\DLR\\brasil_TDX_"+ n + ".tif"
    tdx = imageio.imread(im)
    tdx = np.asarray(tdx)

#    im5 = r"D:\Works\DLR\\brasil_GCF_2000_"+ n + ".tif"
#    mask = imageio.imread(im5)
#    mask = np.asarray(mask)
#    imgn2 = xsar 
#    del xsar
#    print(np.max(mask))
#    mask = 1 - mask 
#    land_calib = mask <= 0.2
#    nasa_to_xsar = (nasa - xsar)*land_calib
#    for aaa in range(1,5):
#        print("GCF con percentuale compresa tra " + str((aaa )*0.2*100) + " e " + str((aaa + 1)*0.2*100))
#        land_calib1 = mask <= (aaa + 1)*0.2
#        land_calib2 = mask >= (aaa )*0.2
#        land_calibt = land_calib1*land_calib2
#        print(" su un numero di campioni pari a " + str(np.sum(land_calibt)))
#        nasa_to_xsar2 = (nasa - xsar)*land_calibt
#        mean_n_to_x2 = np.sum(nasa_to_xsar2)/(np.sum(land_calibt) + 10**(-10))
#        print(mean_n_to_x2)
#    
#    mean_n_to_x = np.sum(nasa_to_xsar)/(np.sum(land_calib) + 10**(-10))
#    
#    print(mean_n_to_x)
#    
#    nasa = (nasa - mean_n_to_x)*(nasa>0)
#    [A, B] = xsar.shape
#    diff_mean = []
#    for bb in range(B): 
#        diff_col = tdx[:,bb] - xsar[:,bb]
#        ind = np.nonzero(diff_col)
#        diff_col_nozero2 = [diff_col[kk] for kk in ind[0]]
#        diff_mean.append(np.sum(diff_col_nozero2)/len(diff_col_nozero2))
#
#    fimg3_T = np.ndarray(shape=(0,B), dtype='float32')
#    fimg21 = np.reshape(diff_mean, newshape = (1,B))
##    fimg21 = diff_mean
#    for k in range(A):
#        fimg3_T = np.concatenate((fimg3_T, fimg21))
#    imgn2 = (xsar + fimg3_T)*(xsar>0)
#    
#    mask2 = xsar == 0
##    mask = diff == 0
    mask = loss > 15# loss == 0 and loss > 15 
    [s1, s2] = nasa.shape
    print(nasa.shape)
    p2 = []
    for y in range(1,s1-ps+1,r): 
        for x in range(1,s2-ps+1,r):
##            mask_d0 = loss[y:y+ps,x:x+ps]
            mask_d1 = mask[y:y+ps,x:x+ps]
#            [m1,m2] = mask_d1.shape
##            s_0 =  mask_d0.sum()
            s_1 =  mask_d1.sum()
            if s_1 == 0: #s_0 == 0 and 
                p2.append([y,x])
    p = p2
    random.shuffle(p)
    print(len(p2))
    P = len(p) 
    p_train,p_val= p[:int(0.8*P)],p[int(0.8*P):P]
    
#### 1 ######

    x_train_k = np.ndarray(shape=(len(p_train), ps,ps, N), dtype='float32')
    y_train_k = np.ndarray(shape=(len(p_train), ps,ps, Out), dtype='float32')
    print(y_train.shape)
    n1 = 0
    for patch in p_train:
#        print(n1)
        y0, x0 = patch[0], patch[1]
        x_train_k[n1,:,:,0] = nasa[y0:y0+ps,x0:x0+ps]/1000 # - tdx[y0:y0+ps,x0:x0+ps]/1000
#        x_train_k[n1,:,:,1] = mask[y0:y0+ps,x0:x0+ ps]
        x_train_k[n1,:,:,1] = tdx[y0:y0+ps,x0:x0+ ps]/1000
#        x_train_k[n1,:,:,3] = land15[y0:y0+ps,x0:x0+ ps]
        y_train_k[n1,:,:,0] = loss[y0:y0+ps,x0:x0+ ps] > 0
#        y_train_k[n1,:,:,0]= imgn2[y0:y0+ps,x0:x0+ ps]/1000 - tdx[y0:y0+ps,x0:x0+ps]/1000#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#        y_train_k[n1,:,:,1]= imgn2[y0:y0+ps,x0:x0+ ps]/1000 - nasa[y0:y0+ps,x0:x0+ps]/1000 #-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#        y_train_k[n1,:,:,2]= mask[y0:y0+ps,x0:x0+ ps] 
        n1 += 1
    x_train = np.concatenate((x_train, x_train_k))
    y_train = np.concatenate((y_train, y_train_k))
    print(np.max(x_train))
    x_val_k = np.ndarray(shape=(len(p_val), ps,ps, N), dtype='float32')
    y_val_k = np.ndarray(shape=(len(p_val), ps,ps, Out), dtype='float32')
    
    n1 = 0
    for patch in p_val:
        y0, x0 = patch[0], patch[1]
        x_val_k[n1,:,:,0] = nasa[y0:y0+ps,x0:x0+ps]/1000 #- tdx[y0:y0+ps,x0:x0+ps]/1000
#        x_val_k[n1,:,:,1] = mask[y0:y0+ps,x0:x0+ ps]
        x_val_k[n1,:,:,1] = tdx[y0:y0+ps,x0:x0+ ps]/1000
#        x_val_k[n1,:,:,3] = land15[y0:y0+ps,x0:x0+ ps]
        y_val_k[n1,:,:,0] = loss[y0:y0+ps,x0:x0+ ps]> 0
#        y_val_k[n1,:,:,0] = imgn2[y0:y0+ps,x0:x0+ ps] - tdx[y0:y0+ps,x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#        y_val_k[n1,:,:,1] = imgn2[y0:y0+ps,x0:x0+ ps] - nasa[y0:y0+ps,x0:x0+ps]
#        y_val_k[n1,:,:,2] = mask[y0:y0+ps,x0:x0+ ps] 
        n1 += 1
    x_val = np.concatenate((x_val, x_val_k))
    y_val = np.concatenate((y_val, y_val_k))

#### 2 #####

#    x_train_k2 = np.ndarray(shape=(len(p_train), ps,ps, N), dtype='float32')
#    y_train_k2 = np.ndarray(shape=(len(p_train), ps,ps, Out), dtype='float32')
#    
#    n1 = 0
#    for patch in p_train:
#        y0, x0 = patch[0], patch[1]
#        x_train_k2[n1,:,:,0] = nasa[y0:y0+ps,x0:x0+ps]/1000 #- tdx[y0:y0+ps,x0:x0+ps]
#        x_train_k2[n1,:,:,1] = mask[y0:y0+ps,x0:x0+ ps]
#        x_train_k2[n1,:,:,2] = tdx[y0:y0+ps,x0:x0+ ps]/1000
#        x_train_k2[n1,:,:,3] = land15[y0:y0+ps,x0:x0+ ps]
#        y_train_k2[n1,:,:,0]= imgn2[y0:y0+ps,x0:x0+ ps]/1000 #- tdx[y0:y0+ps,x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#        y_train_k2[n1,:,:,1]= imgn2[y0:y0+ps,x0:x0+ ps]/1000 - nasa[y0:y0+ps,x0:x0+ps]/1000 #-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#        y_train_k2[n1,:,:,2]= mask[y0:y0+ps,x0:x0+ ps] 
#        n1 += 1
#    x_train2 = np.concatenate((x_train2, x_train_k2))
#    y_train2 = np.concatenate((y_train2, y_train_k2))
#    print(y_train_k2.shape)
#    print(y_train2.shape)
#
#    x_val_k2 = np.ndarray(shape=(len(p_val), ps,ps, N), dtype='float32')
#    y_val_k2 = np.ndarray(shape=(len(p_val), ps,ps, Out), dtype='float32')
#    
#    n1 = 0
#    for patch in p_val:
#        y0, x0 = patch[0], patch[1]
#        x_val_k2[n1,:,:,0] = nasa[y0:y0+ps,x0:x0+ps]/1000 #- tdx[y0:y0+ps,x0:x0+ps]
#        x_val_k2[n1,:,:,1] = mask[y0:y0+ps,x0:x0+ ps]
#        x_val_k2[n1,:,:,2] = tdx[y0:y0+ps,x0:x0+ ps]/1000
#        x_val_k2[n1,:,:,3] = land15[y0:y0+ps,x0:x0+ ps]
#        y_val_k2[n1,:,:,0] = imgn2[y0:y0+ps,x0:x0+ ps]/1000 #- tdx[y0:y0+ps,x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#        y_val_k2[n1,:,:,1] = imgn2[y0:y0+ps,x0:x0+ ps]/1000 - nasa[y0:y0+ps,x0:x0+ps]/1000
#        y_val_k2[n1,:,:,2] = mask[y0:y0+ps,x0:x0+ ps] 
#        n1 += 1
#    x_val2 = np.concatenate((x_val2, x_val_k2))
#    y_val2 = np.concatenate((y_val2, y_val_k2))
#
#    ndvi3 = np.asarray(imgn2)
#    im3 = r"D:\Works\DLR\brasil_D_XSAR_"+ n + ".tif"
#    imsave(im3, ndvi3)
#
#    ndvi4 = np.asarray(nasa)
#    im4 = r"D:\Works\DLR\brasil_C_NASADEM_"+ n + ".tif"
#    imsave(im4, ndvi4)
#  
#    ndvi4 = np.asarray(tdx)
#    im4 = r"D:\Works\DLR\brasil_TDX2_"+ n + ".tif"
#    imsave(im4, ndvi4)

np.savez("train_data.npz", x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val)    
#np.savez("train_data2.npz", x_train = x_train2, y_train = y_train2, x_val = x_val2, y_val = y_val2)    

    