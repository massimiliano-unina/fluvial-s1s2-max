# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:12:24 2019

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
from scipy.ndimage import gaussian_filter, morphology,generate_binary_structure
import cmath 
from skimage import filters
from main_functions import * 

#N_Images = 5
#mean_diff = np.zeros(255)
#X4 = []
#TX4 = []
#N4 = []
#LAND4 = []
#DIFF4 = []
#DIFFXTX = []
    
imgs2 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_SRTM.tif")
imgx2 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_XSAR.tif")
imgn2 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_NASADEM.tif")
imgtx2 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_TDX.tif")
land = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_landsat_treecover.tif")

diffxtx = imgx2 - imgn2 
diff_land_zero = diffxtx*(land == 0)
meanx = np.sum(diff_land_zero)/(np.sum(land == 0) + 10**(-10))

imgx3 = (imgx2 - meanx)*( 1 - imgx2 == 0)

ndvi3 = np.asarray(imgx3)
im3 = r"D:\Works\DLR\Uganda_Solberg\\uganda_XSAR_noshift.tif"
imsave(im3, ndvi3)

