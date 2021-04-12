# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:57:21 2019

@author: massi
"""

import numpy as np 
from tifffile import imsave

folder = r"D:\Works\Albufera-SemanticSegmentation\S2_S1\\"
size = 128
A = np.ones((3500,2400),dtype='float32')
A[1080:1080 + size,1060:1060 + size] = 0
im = folder + 'Patches_0.tif'
imsave(im, A)

B = np.ones((3500,2400),dtype='float32')
B[1080 + size:1080 + 2*size,1060 + size:1060 + 2*size] = 0
im = folder + 'Patches_1.tif'
imsave(im, B)

B = np.ones((3500,2400),dtype='float32')
B[1080 + size:1080 + 2*size,1060:1060 + size] = 0
im = folder + 'Patches_2.tif'
imsave(im, B)

B = np.ones((3500,2400),dtype='float32')
B[1080:1080 + size,1060 + size:1060 + 2*size] = 0
im = folder + 'Patches_3.tif'
imsave(im, B)

A = np.ones((3500,2400),dtype='float32')
A[2250:2250 + size,1250:1250 + size] = 0
im = folder + 'Patches_4.tif'
imsave(im, A)