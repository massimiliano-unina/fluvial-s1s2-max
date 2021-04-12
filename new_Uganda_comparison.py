# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:51:28 2019

@author: massi
"""


## This code allow us to consider the unchanged zone between 2000 and 2015 in \" + areas + ".
## In particular, we compare the X-band of SRTM mission (2000) and the X-band of TanDEM-X. 
## In order to consider the unchanged zone we use the GLAD Global forest cover loss 
## from 2000 to 2017, so we are sure to consider unchanged in 2000-2015 period. 

import imageio 
import numpy as np
from skimage.transform import resize
from tifffile import imsave 
from matplotlib import pyplot as plt

path1 = [r"D:\Works\DLR\Landcover\\", r"D:\Works\DLR\Uganda_Solberg\\"]
areas1 = ["brasil", "uganda"]

#for different in range(2):
different = 1
path = path1[different]
areas = areas1[different]
print("opening NASA")
nasa = imageio.imread(path + areas + "_NASADEM.tif")
if different == 1: 
    nasa = nasa[:-1, :-1]
print("DONE NASA!!")

nasa2 = np.zeros(shape=(int(nasa.shape[0]/4), int(nasa.shape[1]/4)))
a = [(0,0), (0,1),(0,2),(0,3), (1,0), (1,1), (1,2), (1,3),(2,0), (2,1), (2,2),(2,3),(3,0), (3,1), (3,2),(3,3) ]
c = 0 
for n in a: 
    nasa2 = nasa[n[0]*int(nasa.shape[0]/4):(n[0] + 1)*int(nasa.shape[0]/4), n[1]*int(nasa.shape[1]/4):(n[1] + 1)*int(nasa.shape[1]/4)]
    ndvi3 = np.asarray(nasa2)
    im3 = r"D:\Works\DLR\\" + areas + "_NASADEM_"+ str(n[0]) + "_"+ str(n[1]) + ".tif"
    imsave(im3, ndvi3)
    c += 1
    print("save NASA " + str(c/len(a)))   

del nasa
del nasa2
del ndvi3

print("opening XSAR!!")
xsar = imageio.imread(path + areas + "_XSAR.tif")
if different == 1: 
    xsar = xsar[:-1, :-1]
print("DONE XSAR!!")

mask_xsar = np.invert(xsar == 0)
print("Mask XSAR!!")

xsar2 = np.zeros(shape=(int(xsar.shape[0]/4), int(xsar.shape[1]/4)))
a = [(0,0), (0,1),(0,2),(0,3), (1,0), (1,1), (1,2), (1,3),(2,0), (2,1), (2,2),(2,3),(3,0), (3,1), (3,2),(3,3) ]
c = 0
for n in a: 
    xsar2 = xsar[n[0]*int(xsar.shape[0]/4):(n[0] + 1)*int(xsar.shape[0]/4), n[1]*int(xsar.shape[1]/4):(n[1] + 1)*int(xsar.shape[1]/4)]
    ndvi3 = np.asarray(xsar2)
    im3 = r"D:\Works\DLR\\" + areas + "_XSAR_"+ str(n[0]) + "_"+ str(n[1]) + ".tif"
    imsave(im3, ndvi3)
    c += 1
    print("save XSAR " + str(c/len(a)))   

del xsar2 
del xsar 
del ndvi3

print("opening TDX!!")
tdx = imageio.imread(path + areas + "_TDX.tif")
if different == 1: 
    tdx = tdx[:-1, :-1]
print("DONE TDX!!")

mask_tdx = np.invert(tdx == -32767)
print("Mask TDX!!")

mask_x_tdx = mask_xsar*mask_tdx 
del mask_tdx 
del mask_xsar

tdx2 = np.zeros(shape=(int(tdx.shape[0]/4), int(tdx.shape[1]/4)))
a = [(0,0), (0,1),(0,2),(0,3), (1,0), (1,1), (1,2), (1,3),(2,0), (2,1), (2,2),(2,3),(3,0), (3,1), (3,2),(3,3) ]
c = 0
for n in a: 

    tdx2 = tdx[n[0]*int(tdx.shape[0]/4):(n[0] + 1)*int(tdx.shape[0]/4), n[1]*int(tdx.shape[1]/4):(n[1] + 1)*int(tdx.shape[1]/4)]
    ndvi3 = np.asarray(tdx2)
    im3 = r"D:\Works\DLR\\" + areas + "_TDX_"+ str(n[0]) + "_"+ str(n[1]) + ".tif"
    imsave(im3, ndvi3)
    c += 1
    print("save TDX " + str(c/len(a)))   

del tdx2 
del tdx  
del ndvi3

print("opening LANDSAT!!")
land_20152 = imageio.imread(path + areas + "_landsat_2015.tif")
if different == 1: 
    land_2015 = land_20152[:-1, :-1]
else: 
    land_2015 = land_20152#[:-1, :-1]
print("DONE LANDSAT!!")
del land_20152

land_2015 = (land_2015 - np.min(land_2015))/(np.max(land_2015) - np.min(land_2015))
mask_land_2015 = np.invert(land_2015 == 0)
print("Mask LANDSAT!!")

mask_l_x_tdx = mask_x_tdx*mask_land_2015
del mask_x_tdx 
del mask_land_2015

MM = land_2015.shape
nasa2 = np.zeros(shape=(int(MM[0]/4), int(MM[1]/4)))
a = [(0,0), (0,1),(0,2),(0,3), (1,0), (1,1), (1,2), (1,3),(2,0), (2,1), (2,2),(2,3),(3,0), (3,1), (3,2),(3,3) ]
c = 0
for n in a: 
    nasa2 = land_2015[n[0]*int(MM[0]/4):(n[0] + 1)*int(MM[0]/4), n[1]*int(MM[1]/4):(n[1] + 1)*int(MM[1]/4)]
    ndvi3 = np.asarray(nasa2)
    im3 = r"D:\Works\DLR\\" + areas + "_landsat_2015_"+ str(n[0]) + "_"+ str(n[1]) + ".tif"
    imsave(im3, ndvi3)
    c += 1
    print("save LANDSAT " + str(c/len(a)))   
    
del land_2015
del nasa2
del ndvi3

##############

print("opening GLAD!!")

glad = imageio.imread(path + areas + "_GLAD_2017_2000.tif")
glad2 = resize(glad, output_shape = MM, order = 1, mode = 'constant')
print("resized GLAD!!")

del glad
print("opening LAND 2000!!")

mask_glad = (glad2 == -9000) + (glad2 > 0)
mask_glad = np.invert(mask_glad > 0) 
print("Mask GLAD!!")
mask_g_l_x_tdx = mask_l_x_tdx*mask_glad
del mask_l_x_tdx 
del mask_glad

land = imageio.imread(path + areas + "_GCF_2000.tif")
land2 = resize(land, output_shape = MM, order = 1, mode = 'constant')
land2 = (land2 - np.min(land2))/(np.max(land2) - np.min(land2))

print("resized and stretched LAND 2000!!")

del land



nasa2 = np.zeros(shape=(int(glad2.shape[0]/4), int(glad2.shape[1]/4)))
a = [(0,0), (0,1),(0,2),(0,3), (1,0), (1,1), (1,2), (1,3),(2,0), (2,1), (2,2),(2,3),(3,0), (3,1), (3,2),(3,3) ]
c = 0 
for n in a: 
    nasa2 = glad2[n[0]*int(glad2.shape[0]/4):(n[0] + 1)*int(glad2.shape[0]/4), n[1]*int(glad2.shape[1]/4):(n[1] + 1)*int(glad2.shape[1]/4)]
    ndvi3 = np.asarray(nasa2)
    im3 = r"D:\Works\DLR\\" + areas + "_GLAD_2017_2000_"+ str(n[0]) + "_"+ str(n[1]) + ".tif"
    imsave(im3, ndvi3)
    c += 1 
    print("save GLAD " + str(c/len(a)))   

del nasa2
del glad2
del ndvi3

mask_land = np.invert(land2 == 0)
print("Mask LAND 2000!!")
mask_tot = mask_g_l_x_tdx*mask_land
del mask_g_l_x_tdx 
del mask_land

nasa2 = np.zeros(shape=(int(land2.shape[0]/4), int(land2.shape[1]/4)))
a = [(0,0), (0,1),(0,2),(0,3), (1,0), (1,1), (1,2), (1,3),(2,0), (2,1), (2,2),(2,3),(3,0), (3,1), (3,2),(3,3) ]
c = 0
for n in a: 
    nasa2 = land2[n[0]*int(land2.shape[0]/4):(n[0] + 1)*int(land2.shape[0]/4), n[1]*int(land2.shape[1]/4):(n[1] + 1)*int(land2.shape[1]/4)]
    ndvi3 = np.asarray(nasa2)
    im3 = r"D:\Works\DLR\\" + areas + "_GCF_2000_"+ str(n[0]) + "_"+ str(n[1]) + ".tif"
    imsave(im3, ndvi3)
    c += 1 
    print("save LAND 2000 " + str(c/len(a)))   

del nasa2
del land2
del ndvi3

nasa2 = np.zeros(shape=(int(mask_tot.shape[0]/4), int(mask_tot.shape[1]/4)))
a = [(0,0), (0,1),(0,2),(0,3), (1,0), (1,1), (1,2), (1,3),(2,0), (2,1), (2,2),(2,3),(3,0), (3,1), (3,2),(3,3) ]
c = 0 
for n in a: 
    nasa2 = mask_tot[n[0]*int(mask_tot.shape[0]/4):(n[0] + 1)*int(mask_tot.shape[0]/4), n[1]*int(mask_tot.shape[1]/4):(n[1] + 1)*int(mask_tot.shape[1]/4)]
    ndvi3 = np.asarray(nasa2)
    im3 = r"D:\Works\DLR\\" + areas + "_Valid_"+ str(n[0]) + "_"+ str(n[1]) + ".tif"
    imsave(im3, ndvi3)
    c += 1
    print("save Mask Total " + str(c/len(a)))   

del nasa2
del mask_tot
del ndvi3
    
