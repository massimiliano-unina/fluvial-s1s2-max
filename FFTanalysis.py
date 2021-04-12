# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:39:00 2019

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

import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data
from skimage import filters
from main_functions import * 

for kkk1 in range(4):
    
#    "D:\Works\DLR\Landcover\brasil_nasadem.tif"
#"D:\Works\DLR\Landcover\brasil_xsar.tif"
    imgtx1 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_TDX_daruotare"+str(kkk1 +1)+"_fatto.tif")
    imgx1 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_XSAR_daruotare"+str(kkk1 +1)+"_fatto.tif")
    imgn1 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_NASADEM_daruotare"+str(kkk1 +1)+"_fatto.tif")
    land = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_landcover_daruotare"+str(kkk1 +1)+"_fatto.tif")
    Land = np.mean(land, axis = 0)
    
    imgx = np.asarray(imgx1)
    imgn = np.asarray(imgn1)
    imgtx = np.asarray(imgtx1)
    imgx2 = np.asarray(imgx1)
    imgn2 = np.asarray(imgn1)
    imgtx2 = np.asarray(imgtx1)
    imgxa = np.mean(imgx, axis = 0)
    imgtxb = np.mean(imgtx, axis = 0)
    err_line = (imgxa - np.mean(imgx)) - (imgtxb - np.mean(imgtx))
#    plt.figure()
#    plt.plot(err_line)

    [A, B] = imgx.shape

    fimg3_T = func_1d_2_column2d(err_line)
    fimg3_T = gaussian_filter(fimg3_T, sigma=(5,5))
### Tentativo Inutile     
    imgxTh1 = imgx #np.zeros(shape=(land.shape[0],land.shape[1]))
    imgnTh2 = imgtx # np.zeros(shape=(land.shape[0],land.shape[1]))
    for Land_thresh in range(1):
#        print(Land_thresh)
#        land_Th1 = np.zeros(shape=(land.shape[0],land.shape[1]))
        land_Th1 = land == Land_thresh #land > (Land_thresh - 1)*0.1*255
#        land_Th2 = land < (Land_thresh)*0.1*255
#        land_Th[:,:,Land_thresh] = land_Th1#*land_Th2
#        land_list.append((np.sum(land*land_Th[:,:,Land_thresh]) + 10**(-10))/(np.count_nonzero(land_Th[:,:,Land_thresh]) + 10**(-10)))
        if np.count_nonzero(land_Th1) != 0: 
#            print(((np.sum(imgn*land_Th1))/np.count_nonzero(land_Th1))*land_Th1)
#            print(np.sum(imgx*land_Th1)/np.count_nonzero(land_Th1))
#            print(np.sum(imgx*land_Th1))
            imgxTh1 = imgxTh1*(1 - land_Th1) + imgnTh2*land_Th1 # - ((np.sum((imgx - imgn)*land_Th1))/np.count_nonzero(land_Th1))*land_Th1
            imgnTh2 = imgnTh2 #- ((np.sum((imgx - imgn)*land_Th1))/np.count_nonzero(land_Th1))*land_Th1
#            print(type(imgxTh1))
#            print(type(imgx))
#
#    ndvi3 = np.asarray(imgxTh1)
#    im3 = r"D:\Works\DLR\Uganda_Solberg\\XSAR_"+str(kkk1 +1)+".tif"
#    imsave(im3, ndvi3)
#
#
#    ndvi3 = np.asarray(imgnTh2)
#    im3 = r"D:\Works\DLR\Uganda_Solberg\\NASADEM_"+str(kkk1 +1)+".tif"
#    imsave(im3, ndvi3)
#




## Wavelet transform of image, and plot approximation and details
#    titles = ['Approximation', ' Horizontal detail',
#              'Vertical detail', 'Diagonal detail']
#    coeffs2 = pywt.dwt2(imgx - imgn, 'bior1.3')
#    LL, (LH, HL, HH) = coeffs2
#    
#    coeffs2n = pywt.dwt2(imgn, 'bior1.3')
#    LLn, (LHn, HLn, HHn) = coeffs2n
#    fig = plt.figure(figsize=(12, 3))
#    for i, a in enumerate([LL, LH, HL, HH]):
#        ax = fig.add_subplot(1, 4, i + 1)
#        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#        ax.set_title(titles[i], fontsize=10)
#        ax.set_xticks([])
#        ax.set_yticks([])
#        ndvi3 = np.asarray(a)
#        im3 = r"D:\Works\DLR\Uganda_Solberg\\XSAR_"+ titles[i] +"_" +str(kkk1 +1)+".tif"
#        imsave(im3, ndvi3)
#
#
#
##    for i, a in enumerate([LLn, LHn, HLn, HHn]):
##        ax = fig.add_subplot(1, 4, i + 1)
##        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
##        ax.set_title(titles[i], fontsize=10)
##        ax.set_xticks([])
##        ax.set_yticks([])
##        ndvi3 = np.asarray(a)
##        im3 = r"D:\Works\DLR\Uganda_Solberg\\NASADEM_"+ titles[i] +"_" +str(kkk1 +1)+".tif"
##        imsave(im3, ndvi3)
##
##    fig.tight_layout()
##    plt.show()
##
#    edges_x = filters.sobel_h(imgx - imgn) 
#    edges_y = filters.sobel_v(imgx - imgn)
#    edges = filters.sobel(imgx - imgn)
#    
#    ndvi3 = np.asarray(edges_x)
#    im3 = r"D:\Works\DLR\Uganda_Solberg\\SobelX_XSAR_"+ titles[i] +"_" +str(kkk1 +1)+".tif"
#    imsave(im3, ndvi3)
#
#    ndvi3 = np.asarray(edges_y)
#    im3 = r"D:\Works\DLR\Uganda_Solberg\\SobelY_XSAR_"+ titles[i] +"_" +str(kkk1 +1)+".tif"
#    imsave(im3, ndvi3)
#
#    ndvi3 = np.asarray(edges)
#    im3 = r"D:\Works\DLR\Uganda_Solberg\\Sobel_XSAR_"+ titles[i] +"_" +str(kkk1 +1)+".tif"
#    imsave(im3, ndvi3)
#
#
#    fig.tight_layout()
#    plt.show()
#
    
    
#    Land = gaussian_filter(Land, sigma=25)
#    
#    imgx_line = np.mean(imgx, axis = 0)
#    imgtx_line = np.mean(imgtx, axis = 0)
#    imgn_line = np.mean(imgn, axis = 0)
#    
#    imgxS = imgx - np.mean(imgx)
#    imgnS = imgn - np.mean(imgn)
#    
#    err = imgxS - imgnS 
#    err_line = np.mean(err, axis = 0)
#    
#    [A, B] = imgx.shape
#
#    fimg3_T = np.ndarray(shape=(0,B), dtype='float32')
#    fimg21 = np.reshape(err_line, newshape = (1,B))
#    
#    for k in range(A):
#        fimg3_T = np.concatenate((fimg3_T, fimg21))
#    
#    err2 = err - fimg3_T 
    


    
#    imgx2 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_XSAR_DeNoise_daruotare_fatto_FFT2.tif")
#    imgn2 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_NASADEM_DeNoise_daruotare_fatto_FFT2.tif")
    
#    IM = imgx2 - imgn2
#    IMM = np.mean(IM, axis = 0)
#    
#    plt.figure()
#    #t.subplot(311)
#    plt.plot(IMM)#, cmap='gray')
    #
    #
    ##D = land1 == np.min(land1)
    ##Diff1 = imgx1 - imgn1
    ##Diff2 = Diff1*D
    ##undesired_bias = np.count_nonzero(Diff2)/np.count_nonzero(D)
    ##print(undesired_bias)
    ##imgx11 = imgx1 - Diff2
    #
    #


    f33 = np.fft.fft2(fimg3_T)
    fshiftrumore = np.fft.fftshift(f33)
    rumore = 20*np.log(np.abs(fshiftrumore))
    rumore, fshiftrumore = compute_fft(imgx)

    magnitude_spectrum1, fshift1 = compute_fft(imgx)
    magnitude_spectrum1err, fshift1e = compute_fft(imgx - imgn)

    magnitude_spectrum2, fshift2 = compute_fft(imgn)
    

    magnitude_spectrum3, fshift3 = compute_fft(imgtx)
    magnitude_spectrum1e, fshift11 = compute_fft(imgx)
    magnitude_spectrum2e, fshift21 = compute_fft(imgn)


    
    
    print(type(magnitude_spectrum1))
    magnitude_spectrum11 = np.asarray(fshift1)
    magnitude_spectrum11err = np.asarray(fshift1e)
    magnitude_spectrum21 = np.asarray(fshift2)
    magnitude_spectrum31 = np.asarray(fshift3)
    magnitude_spectrum11_r = gaussian_filter(np.real(magnitude_spectrum11), sigma=(10,10))
    magnitude_spectrum11_i = gaussian_filter(np.imag(magnitude_spectrum11), sigma=(10,10))
    
    magnitude_spectrum12 = np.zeros(magnitude_spectrum11.shape,dtype='complex')
#    for x in range(magnitude_spectrum11.shape[0]):
#        for y in range(magnitude_spectrum11.shape[1]):
#            magnitude_spectrum12[x,y] = np.complex(magnitude_spectrum11_r[x,y], magnitude_spectrum11_i[x,y])
    W = 1500 
    L = 2
    for kf in range(1):
        filter11 = np.zeros(magnitude_spectrum11.shape)
#    
#    #    filter11[ magnitude_spectrum11.shape[0]//2 - 10*L:magnitude_spectrum11.shape[0]//2 +10*L, magnitude_spectrum11.shape[1]//2 - W: magnitude_spectrum11.shape[1]//2 -L] = 0
#    #    filter11[ magnitude_spectrum11.shape[0]//2 - 10*L:magnitude_spectrum11.shape[0]//2 +10*L, magnitude_spectrum11.shape[1]//2 + L : magnitude_spectrum11.shape[1]//2 + W] = 0
#        filter11[ : magnitude_spectrum11.shape[0]//2 -1, magnitude_spectrum11.shape[1]//2 ] = 0
#        filter11[ magnitude_spectrum11.shape[0]//2 +1: , magnitude_spectrum11.shape[1]//2 ] = 0
#    #    filter11[imgn.shape[0]//2 - W1:imgn.shape[0]//2 - L1, imgn.shape[1]//2 - W:imgn.shape[1]//2 + W] = 0
#    #    filter11[imgn.shape[0]//2 + L1:imgn.shape[0]//2+W1, imgn.shape[1]//2 - W:imgn.shape[1]//2 + W] = 0
#    #    filter11[imgn.shape[0]//2 - W1:imgn.shape[0]//2+W1, imgn.shape[1]//2 - W:imgn.shape[1]//2 - L] = 0
#    #    filter11[imgn.shape[0]//2 - W1:imgn.shape[0]//2+W1, imgn.shape[1]//2 + L:imgn.shape[1]//2 + W] = 0
#
#        filter11[ magnitude_spectrum11.shape[0]//2 - 10:magnitude_spectrum11.shape[0]//2 + 10 , magnitude_spectrum11.shape[1]//2 - 10:magnitude_spectrum11.shape[1]//2 + 10] = 1
#        filter11 = gaussian_filter(filter11, sigma=(25,25))

        filter11[:, imgn.shape[1]//2 - 20:imgn.shape[1]//2 + 20] = 1
        filter11[imgn.shape[0]//2 - 20 :imgn.shape[0]//2 + 20, :] = 1
        filter11 = gaussian_filter(filter11, sigma=(25,25))
#        filter11 = rumore 
        #        filter12[imgn.shape[0]//2 - 1:imgn.shape[0]//2 + 1, :] = 1
    ##        filter12[imgn.shape[0]//2 , :] = 1
    #        filter12[:,imgn.shape[1]//2 - 3:imgn.shape[1]//2 + 3] = 1
    #        filter12 = gaussian_filter(filter12, sigma=(1.5,2.5))
        
        magnitude_spectrum11 = magnitude_spectrum11*filter11
        magnitude_spectrum11 = magnitude_spectrum11*filter11
        magnitude_spectrum21 = magnitude_spectrum21*filter11
        magnitude_spectrum31 = magnitude_spectrum31*filter11
    #    filter22 = np.ones(magnitude_spectrum11.shape)
        
    #        magnitude_spectrum11[imgn.shape[0]//2 - 10:imgn.shape[0]//2 + 10, imgn.shape[1]//2 -10:imgn.shape[1]//2 + 10]  = fshift11[imgn.shape[0]//2 - 10:imgn.shape[0]//2 + 10, imgn.shape[1]//2 -10:imgn.shape[1]//2 + 10]
    #        magnitude_spectrum21[imgn.shape[0]//2 - 10:imgn.shape[0]//2 + 10, imgn.shape[1]//2 -10:imgn.shape[1]//2 + 10] = fshift21[imgn.shape[0]//2 - 10:imgn.shape[0]//2 + 10, imgn.shape[1]//2 -10:imgn.shape[1]//2 + 10]
    #        magnitude_spectrum31[imgn.shape[0]//2 - 10:imgn.shape[0]//2 + 10, imgn.shape[1]//2 -10:imgn.shape[1]//2 + 10] = fshift31[imgn.shape[0]//2 - 10:imgn.shape[0]//2 + 10, imgn.shape[1]//2 -10:imgn.shape[1]//2 + 10]
    
    
        
    #        magnitude_spectrum11[imgn.shape[0]//2 - W1:imgn.shape[0]//2 - L1, imgn.shape[1]//2 - W:imgn.shape[1]//2 + W] = 0
    #        magnitude_spectrum21[imgn.shape[0]//2 - W1:imgn.shape[0]//2 - L1, imgn.shape[1]//2 - W:imgn.shape[1]//2 + W] = 0
    #        magnitude_spectrum31[imgn.shape[0]//2 - W1:imgn.shape[0]//2 - L1, imgn.shape[1]//2 - W:imgn.shape[1]//2 +W] = 0
    #        magnitude_spectrum11[imgn.shape[0]//2 + L1:imgn.shape[0]//2+W1, imgn.shape[1]//2 - W:imgn.shape[1]//2 + W] = 0
    #        magnitude_spectrum21[imgn.shape[0]//2 + L1:imgn.shape[0]//2+W1, imgn.shape[1]//2 - W:imgn.shape[1]//2 + W] = 0
    #        magnitude_spectrum31[imgn.shape[0]//2 + L1:imgn.shape[0]//2+W1, imgn.shape[1]//2 - W:imgn.shape[1]//2 + W] = 0
    #        
    #        magnitude_spectrum11[imgn.shape[0]//2 - W1:imgn.shape[0]//2+W1, imgn.shape[1]//2 - W:imgn.shape[1]//2 - L] = 0
    #        magnitude_spectrum21[imgn.shape[0]//2 - W1:imgn.shape[0]//2+W1, imgn.shape[1]//2 - W:imgn.shape[1]//2 - L] = 0
    #        magnitude_spectrum31[imgn.shape[0]//2 - W1:imgn.shape[0]//2+W1, imgn.shape[1]//2 - W:imgn.shape[1]//2 - L] = 0
    #        magnitude_spectrum11[imgn.shape[0]//2 - W1:imgn.shape[0]//2+W1, imgn.shape[1]//2 + L:imgn.shape[1]//2 + W] = 0
    #        magnitude_spectrum21[imgn.shape[0]//2 - W1:imgn.shape[0]//2+W1, imgn.shape[1]//2 + L:imgn.shape[1]//2 + W] = 0
    #        magnitude_spectrum31[imgn.shape[0]//2 - W1:imgn.shape[0]//2+W1, imgn.shape[1]//2 + L:imgn.shape[1]//2 + W] = 0
    
    
        plt.figure()
        plt.subplot(511)
        plt.imshow(20*np.log10(np.abs(magnitude_spectrum11)),cmap='gray')
        plt.subplot(512)
        plt.imshow(20*np.log10(np.abs(magnitude_spectrum21)),cmap='gray')
        plt.subplot(513)
        plt.imshow(20*np.log10(np.abs(magnitude_spectrum31)),cmap='gray')
        plt.subplot(514)
        plt.imshow(filter11,cmap='gray')
        plt.subplot(515)
        plt.imshow(rumore,cmap='gray')
#    
    
        fimg1_T = np.abs(np.fft.ifft2(np.fft.ifftshift(magnitude_spectrum11)))
        fimg2_T = np.abs(np.fft.ifft2(np.fft.ifftshift(magnitude_spectrum21)))
        fimg3_T = np.abs(np.fft.ifft2(np.fft.ifftshift(magnitude_spectrum31)))
#    
#    
        ndvi3 = np.asarray(fimg1_T)#imgx - fimg3_T)
        im3 = r"D:\Works\DLR\Uganda_Solberg\\XDenoised_"+str(kkk1 +1)+"_"+str(kf +1)+".tif"
        imsave(im3, ndvi3)
    
#        ndvi3 = np.asarray(fimg2_T)
#        im3 = r"D:\Works\DLR\Uganda_Solberg\\NDenoised_"+str(kkk1 +1)+"_"+str(kf +1)+".tif"
#        imsave(im3, ndvi3)
#    
#    
#        ndvi3 = np.asarray(fimg3_T)
#        im3 = r"D:\Works\DLR\Uganda_Solberg\\TXDenoised_"+str(kkk1 +1)+"_"+str(kf +1)+".tif"
#        imsave(im3, ndvi3)


#        ndvi3 = np.asarray(fimg3_T)
#        im3 = r"D:\Works\DLR\Uganda_Solberg\\"+str(band +1)+"_TX_FFT"+str(kkk1 +1)+".tif"
#        imsave(im3, ndvi3)
#        magnitude_spectrum11 = m11
#        magnitude_spectrum21 = m21
#        magnitude_spectrum31 = m31
#        
#        plt.figure()
#        plt.subplot(311)
#        plt.imshow(np.abs(magnitude_spectrum11),cmap='gray')
#        plt.subplot(312)
#        plt.imshow(np.abs(magnitude_spectrum21),cmap='gray')
#        plt.subplot(313)
#        plt.imshow(np.abs(magnitude_spectrum31),cmap='gray')
    
