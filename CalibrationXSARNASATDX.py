# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:25:54 2019

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


imgtx = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\line_uganda_TDX.tif")
imgx = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_XSAR_noerr3.tif")
imgn = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\line_uganda_NASADEM.tif")
land = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\line_uganda_landcover_tree.tif")

imgX = np.asarray(imgx)
imgN = np.asarray(imgn)
imgTX = np.asarray(imgtx)
Land = np.asarray(land)

D = Land == np.min(Land)
Diff = imgX - imgN
Diff2 = Diff*D


undesired_bias = np.count_nonzero(Diff2)/np.count_nonzero(D)
imgX1 = imgX - Diff2

ndvi4 = imgX1.astype('float32')
im4 =  r"D:\Works\DLR\Uganda_Solberg\line_uganda_XSAR_noerr3_minus_bias.tif"
imsave(im4, ndvi4)

ndvi4 = Diff.astype('float32')
im4 =  r"D:\Works\DLR\Uganda_Solberg\line_uganda_diff_XSAR_CSAR.tif"
imsave(im4, ndvi4)
