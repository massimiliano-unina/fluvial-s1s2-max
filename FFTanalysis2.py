# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:33:27 2019

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
from main_functions import * 

for kkk1 in range(4):
    imgtx1 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_TDX_daruotare"+str(kkk1 +1)+"_fatto.tif")
    imgx1 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_XSAR_daruotare"+str(kkk1 +1)+"_fatto.tif")
    imgn1 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_NASADEM_daruotare"+str(kkk1 +1)+"_fatto.tif")
    land = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_landcover_daruotare"+str(kkk1 +1)+"_fatto.tif")
    Land = np.mean(land, axis = 0)
    land_Th = np.zeros(shape=(land.shape[0],land.shape[1], 10))

    imgtx = np.asarray(imgtx1)
    imgx = np.asarray(imgx1)
    imgn = np.asarray(imgn1)
    land = np.asarray(land)
    mean_X = np.mean(imgx)
    mean_TX = np.mean(imgtx)
    mean_N = np.mean(imgn)
    imgx1 = imgx - mean_X
    imgn1 = imgn - mean_N
    imgtx1 = imgtx - mean_TX
    
    img = imgtx1 - imgx1 #- imgn 
    imgN = imgtx1 - imgn1 
    [A, B] = img.shape
    
    imgx_mean = np.mean(img, axis = 0)
    fshift2, magnitude_spectrum2 = compute_fft(imgx_mean)
    
    imgx_mean1 = gaussian_filter(imgx_mean, sigma=25)
    fshift1, magnitude_spectrum1 = compute_fft(imgx_mean1)
    
    imgx_mean11 = (1/(magnitude_spectrum1 + np.min(magnitude_spectrum1) + 1))*magnitude_spectrum2
    NN = np.count_nonzero(imgx_mean11)
    print(NN)
    imgx_mean12 = gaussian_filter(imgx_mean11, sigma=25)
    
    magnitude_to_compensate = 1/(imgx_mean12 + 1)
    magnitude_1 = imgx_mean - imgx_mean1 #imgx_mean11#magnitude_spectrum2*magnitude_to_compensate
    M = imgx_mean.shape
        
    plt.figure()
    plt.subplot(511)
    plt.plot(imgx_mean,'g')
    plt.title('difference')
    plt.subplot(512)
    plt.plot(magnitude_spectrum2,'b')
    plt.title('FFT 1')
    plt.subplot(513)
    plt.plot(imgx_mean1 ,'r')
    plt.title('mean difference')
    plt.subplot(514)
    plt.plot(magnitude_spectrum1,'b')
    plt.title('FFT 2')
    plt.subplot(515)
    plt.plot(magnitude_1,'b')
    plt.title('FFT difference')
    
    fimg2_T = func_1d_2_column2d(imgx_mean1, img)  
    imgx2 = imgx + fimg2_T
    
    imgx_meanN = np.mean(imgN, axis = 0)
    fshift2, magnitude_spectrumN2 = fft1d(imgx_meanN)
    imgx_meanN1 = gaussian_filter(imgx_meanN, sigma=25)
    fshift1, magnitude_spectrumN1 = fft1d(imgx_meanN1)
#    f1 = np.fft.fft(imgx_meanN1)
#    fshift1 = np.fft.fftshift(f1)
#    magnitude_spectrumN1 = 20*np.log(np.abs(fshift1))
    
#    imgx_meanN11 = (1 - magnitude_spectrumN1 > 0)*magnitude_spectrumN2
    imgx_meanN11 = (1/(magnitude_spectrumN1 + np.min(magnitude_spectrumN1) + 1))*magnitude_spectrum2
    NN = np.count_nonzero(imgx_meanN11)
    print(NN)
    imgx_meanN12 = gaussian_filter(imgx_meanN11, sigma=25)
        
    magnitude_to_compensate = 1/(imgx_meanN12 + 1)
    magnitude_2 = imgx_meanN - imgx_meanN1 # imgx_meanN11 # magnitude_spectrumN2*magnitude_to_compensate
    
    M = imgx_meanN.shape
    plt.figure()
    plt.subplot(511)
    plt.plot(imgx_meanN,'g')
    plt.title('difference')
    plt.subplot(512)
    plt.plot(magnitude_spectrumN2,'b')
    plt.title('FFT 1')
    plt.subplot(513)
    plt.plot(imgx_meanN1 ,'r')
    plt.title('mean difference')
    plt.subplot(514)
    plt.plot(magnitude_spectrumN1,'b')
    plt.title('FFT 2')
    plt.subplot(515)
    plt.plot(magnitude_2,'b')
    plt.title('FFT difference')
    
    [A, B] = img.shape
    print(imgx_meanN1.shape)
    
    fimg2_T = func_1d_2_column2d(imgx_meanN1, img)    
    print(fimg2_T.shape)
#    fimg3_T = np.ndarray(shape=(0,M[0]), dtype='float32')
#    fimg21 = np.reshape(imgx_meanN1, newshape = (1,M[0]))
#    
#    for k in range(A):
#        fimg3_T = np.concatenate((fimg3_T, fimg21))
##    fimg2_T = np.abs(np.fft.ifft2(np.fft.ifftshift(fimg2_T)))
##    imgn2 = imgn + fimg2_T
#    imgn2 = imgn + fimg3_T
    
    ##
    ##f1 = np.fft.fft2(img)
    ##fshift1 = np.fft.fftshift(f1)
    ##magnitude_spectrum1 = 20*np.log(np.abs(fshift1))
    ###magnitude_spectrumA = magnitude_spectrum1 - magnitude_gaussian1
    ##
    ##[M, N] = magnitude_spectrum1.shape
    ###print(M[0])
    ##LPF = np.ones(shape=(M,N))
    ##W1 = 500
    ##W2 = 250
    ##LPF[int(M/2) - W2: int(M/2) + W2, int(N/2) - W1: int(N/2) + W1 ] = 1
    ###LPF[int(M/2) - int(W2/5): int(M/2) + int(W2/5), int(N/2) - int(W1/5): int(N/2) + int(W1/5)] = 0
    ##LPF1 = gaussian_filter(LPF, sigma=[int(W2/6), int(W2/4)])
    ###LPF1 = LPF
    ###LPF[int(M/2) - 1: int(M/2) + 1, int(N/2) - 1: int(N/2) + 1] = 0
    ##magnitude_gaussian0 = 1/(np.abs(fshift1) +np.exp(-50))#*(1-LPF1) #gaussian_filter(np.abs(fshift1), sigma=[int(W2/6), int(W2/4)])
    ###magnitude_gaussian3 = -np.angle(fshift1)#*(1-LPF1) # gaussian_filter(np.angle(fshift1), sigma=[int(W2/6), int(W2/4)])#np.angle(fshift1)#
    ##magnitude_gaussian1 = np.ndarray(shape=(M,N), dtype='complex')
    ##
    ##magnitude_gaussian3 = np.zeros(shape=(M,N))#magnitude_gaussian2*(1-LPF1)
    ##
    ##for x in range(M):
    ##    for y in range(N):
    ##        magnitude_gaussian1[x,y] = complex(magnitude_gaussian0[x,y]*np.cos(magnitude_gaussian3[x,y]), magnitude_gaussian0[x,y]*np.sin(magnitude_gaussian3[x,y]))
    ##
    ##H = magnitude_spectrum1*magnitude_gaussian1
    ##
    ###LPF1 = LPF 
    ##print(np.max(LPF1))
    ##print(np.min(LPF1))
    ##
    ###x = np.arange(M[0])
    ###LPF = []
    ###LPF = np.append(LPF,  2*(x[:int(M[0]/2)]/M[0]) )
    ###
    ###LPF = np.append(LPF, + 2 - 2*(x[int(M[0]/2):]/M[0]))
    ###print(LPF[int(M[0]/2)])
    ##magnitude_spectrum2 = np.abs(fshift1)*(1-LPF1)#20*np.log(np.abs(fshift1)*(1-LPF1))
    ##magnitude_spectrum3 = np.abs(fshift1)*(LPF1)#20*np.log(np.abs(fshift1)*(LPF1))
    ##
    ##
    ###fimg_T = np.ndarray(shape=(0,M[0]), dtype='float32')
    ###fimg1 = np.reshape(magnitude_spectrum2, newshape = (1,M[0]))
    ###
    ###fimg2_T = np.ndarray(shape=(0,M[0]), dtype='float32')
    ###fimg21 = np.reshape(magnitude_spectrum3, newshape = (1,M[0]))
    ###
    ##for k in range(A):
    ##    fimg_T = np.concatenate((fimg_T, fimg1))
    ##    fimg2_T = np.concatenate((fimg2_T, fimg21))
    ##
    #fimg_T = np.abs(np.fft.ifft2(np.fft.ifftshift((H))))
    #fimg2_T = np.abs(np.fft.ifft2(np.fft.ifftshift(magnitude_spectrum3)))
    #
    #XSAR = imgtx + fimg_T
    ##fimg3_T = np.abs(np.fft.ifft2(np.fft.ifftshift(magnitude_spectrum3 + magnitude_spectrum2)))
    ##fimg_T = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift1)))
    #print((fimg_T))
    #print((fimg2_T))
    ##print(fimg_T.shape)
    #plt.figure()
    #plt.subplot(411)
    #plt.imshow(img,cmap='gray')
    #plt.subplot(412)
    #plt.imshow(fimg_T, cmap='gray')
    #plt.subplot(413)
    #plt.imshow(fimg2_T, cmap='gray')
    #plt.subplot(414)
    #plt.imshow(LPF1, cmap='gray')
    #
    #plt.figure()
    #plt.subplot(311)
    #plt.imshow(magnitude_spectrum1)
    #plt.subplot(312)
    #plt.imshow(magnitude_spectrum2)
    #plt.subplot(313)
    #plt.imshow(magnitude_spectrum3)
    #
    #
    #
    #
    ##f1 = np.fft.fft2(imgx1)
    ##fshift1 = np.fft.fftshift(f1)
    ##magnitude_spectrum1 = 20*np.log(np.abs(fshift1))
    ##
    ##f2 = np.fft.fft2(imgn)
    ##fshift2 = np.fft.fftshift(f2)
    ##magnitude_spectrum2 = 20*np.log(np.abs(fshift2))
    ##
    ##img = imgtx - imgx #- imgn 
    ##
    ##
    ##img1 = imgx - imgn
    ##f3 = np.fft.fft2(img)
    ##fshift3 = np.fft.fftshift(f3)
    ##magnitude_spectrum3 = 20*np.log(np.abs(fshift3))
    ##
    ##f4 = np.fft.fft2(img1)
    ##fshift4 = np.fft.fftshift(f4)
    ##magnitude_spectrum4 = 20*np.log(np.abs(fshift4))
    ##
    ##fimg = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift3)))
    #
    #ndvi3 = np.asarray(magnitude_spectrum1) #.astype('float32')
    #im3 = r"D:\Works\DLR\Uganda_Solberg\uganda_XSAR_daruotare_fatto_FFT.tif"
    #imsave(im3, ndvi3)
    ##
    ##ndvi3 = np.asarray(magnitude_spectrum2) #.astype('float32')
    ##im3 = r"D:\Works\DLR\Uganda_Solberg\uganda_NASADEM_daruotare_fatto_FFT.tif"
    ##imsave(im3, ndvi3)
    ##
    ##ndvi3 = np.asarray(img) #.astype('float32')
    ##im3 = r"D:\Works\DLR\Uganda_Solberg\uganda_Diff_TDXtoX_daruotare_fatto.tif"
    ##imsave(im3, ndvi3)
    ##
    ##ndvi3 = np.asarray(magnitude_spectrum3) #.astype('float32')
    ##im3 = r"D:\Works\DLR\Uganda_Solberg\uganda_Diff_TDXXtoX_daruotare_fatto_FFT.tif"
    ##imsave(im3, ndvi3)
    ##
#    ndvi3 = np.asarray(imgx2)
#    im3 = r"D:\Works\DLR\Uganda_Solberg\uganda_XSAR_DeNoise_daruotare_fatto_FFT"+str(kkk1 +1)+".tif"
#    imsave(im3, ndvi3)
    save_image(imgx2, r"D:\Works\DLR\Uganda_Solberg\uganda_XSAR_DeNoise_daruotare_fatto_FFT"+str(kkk1 +1)+".tif")
    
    ndvi3 = np.asarray(imgn2)
    im3 = r"D:\Works\DLR\Uganda_Solberg\uganda_NASADEM_DeNoise_daruotare_fatto_FFT"+str(kkk1 +1)+".tif"
    imsave(im3, ndvi3)

    ndvi3 = np.asarray(fimg2_T)
    im3 = r"D:\Works\DLR\Uganda_Solberg\uganda_XSAR_Noise_daruotare_fatto_FFT"+str(kkk1 +1)+".tif"
    imsave(im3, ndvi3)
    
    ndvi3 = np.asarray(fimg3_T)
    im3 = r"D:\Works\DLR\Uganda_Solberg\uganda_NASADEM_Noise_daruotare_fatto_FFT"+str(kkk1 +1)+".tif"
    imsave(im3, ndvi3)

    
    #
    ##ndvi3 = np.asarray(fimg3_T)
    ##im3 = r"D:\Works\DLR\Uganda_Solberg\uganda_NoNoise_daruotare_fatto_FFT.tif"
    ##imsave(im3, ndvi3)
    ##


