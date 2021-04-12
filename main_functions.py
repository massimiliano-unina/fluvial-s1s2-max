# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 07:41:41 2019

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
import cmath 
import numpy as np 
Image.MAX_IMAGE_PIXELS = None
from scipy.ndimage import gaussian_filter, morphology,generate_binary_structure
from skimage import filters

def save_image(x=None, name_file=None):
    ndvi3 = np.asarray(x)#imgx - fimg3_T)
    im3 = name_file
    imsave(im3, ndvi3)
    
def func_1d_2_column2d(imgx_meanN1= None, images = None):
    [A,B] = images.shape
    M = imgx_meanN1.shape
    fimg3_T = np.ndarray(shape=(0,M[0]), dtype='float32')
    fimg21 = np.reshape(imgx_meanN1, newshape = (1,M[0]))
    
    for k in range(A):
        fimg3_T = np.concatenate((fimg3_T, fimg21))
        
        
    return fimg3_T 

def compute_fft(imgx_meanN1=None):
    print('sono nella funzione')
    if len(imgx_meanN1.shape) == 1: 
        f1 = np.fft.fft(imgx_meanN1)
        fshift1 = np.fft.fftshift(f1)
        magnitude_spectrumN1 = 20*np.log(np.abs(fshift1))
    elif len(imgx_meanN1.shape) == 2: 
        print('lunghezza pari a 1')
        print(imgx_meanN1.shape)
        f1 = np.fft.fft2(imgx_meanN1)
        fshift1 = np.fft.fftshift(f1)
        print(fshift1.shape)
        magnitude_spectrumN1 = 20*np.log(np.abs(fshift1))
        print(magnitude_spectrumN1.shape)
        return magnitude_spectrumN1, fshift1

def sobel_gradientM(imgtx3=None):
    slopex3_x = filters.sobel_h(imgtx3) 
    slopex3_y = filters.sobel_v(imgtx3)
   
    return np.sqrt(slopex3_x**2 + slopex3_y**2)

