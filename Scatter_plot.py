# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:10:13 2019

@author: massi
"""
import tensorflow as tf 
from PIL import Image
import numpy as np 
from tifffile import imsave 
import imageio
Image.MAX_IMAGE_PIXELS = None
from keras.layers import Conv2D,BatchNormalization, Input, Add
from keras.models import Sequential,Model 
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import cmath 


for kkk1 in range(1):
    imgtx = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_TDX_daruotare"+str(kkk1 +1)+"_fatto.tif")
    imgx = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_XSAR_daruotare"+str(kkk1 +1)+"_fatto.tif")
    imgn = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_NASADEM_daruotare"+str(kkk1 +1)+"_fatto.tif")#uganda_NASADEM_daruotare"+str(kkk1 +1)+"_fatto.tif")
    land = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_landcover_daruotare"+str(kkk1 +1)+"_fatto.tif")
    
#    plt.figure()
#    plt.subplot(211)
#    plt.imshow(imgx, cmap='gray')
#    plt.subplot(212)
#    plt.imshow(imgn, cmap='gray')
    print(imgx.shape)
    print(land.shape)
    
    Dnx1 = np.asarray(imgx - imgn)
    imgn1 = np.asarray(imgn)
    imgx1 = np.asarray(imgx)
    imgtx1 = np.asarray(imgtx)
    land1 = np.asarray(land)
    Dnx2 = np.reshape(Dnx1, newshape=(1,imgx1.shape[0]*imgx1.shape[1]))   
    imgx2 = np.reshape(imgx1, newshape=(1,imgx1.shape[0]*imgx1.shape[1]))   
    imgtx2 = np.reshape(imgtx1, newshape=(1,imgx1.shape[0]*imgx1.shape[1]))   
    imgn2 = np.reshape(imgn1, newshape=(1,imgx1.shape[0]*imgx1.shape[1]))   
    land2 = np.reshape(land1, newshape=(1,land.shape[0]*land.shape[1]))
    land2 = land2[::1000]
    imgtx2 = imgtx2[::1000]
    imgx2 = imgx2[::1000]
    imgn2 = imgn2[::1000]
    plt.figure()
#    for k in range(10):
#        land_Th1 = land2 > k*(255/10)
#        print('1')
#        land_Th2 = land2 < (k + 1)*(255/10)
#        print('1')
#        land_Th = land_Th1*land_Th2
#        diff_x_tx = (imgx2 - imgtx2)*land_Th 
#        print('1')
#        plt.subplot(10,1,k+1)
#        plt.plot(diff_x_tx)
#        
#    print(Dnx2.shape)
#    print(land2.shape)
#    plt.figure()
#    plt.scatter(land2[:1000:], Dnx2[:1000:])
#    
#    plt.figure()
#    plt.scatter(imgn2[:1000:], imgx2[:1000:])