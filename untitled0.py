# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:40:20 2019

@author: massi
"""

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

nasa = imageio.imread(path + areas + "_NASADEM.tif")
xsar = imageio.imread(path + areas + "_XSAR.tif")
mask = np.invert(xsar == 0)

diff = (xsar - nasa)*mask
MM = diff.shape
del nasa
del xsar 

gcf = imageio.imread(path + areas + "_GCF_2000.tif")
gcf2 = resize(gcf, output_shape = MM, order = 1, mode = 'constant')
del gcf 
diff2 = np.reshape(diff, (1,MM[0]*MM[1]))

tree = []
for n in range(10): 
    level1 = gcf2 > n*10
    level2 = gcf2 < (n+1)*10
    gcf_level = level1*level2*mask
    gcf_level1 = np.reshape(gcf_level, (1,MM[0]*MM[1]))
    g_ind = np.nonzero(gcf_level1)
    diff_nozero = [diff2[0,k] for k in (list(g_ind[1]))]
    tree.append(np.sum(diff_nozero)/np.sum(gcf_level1)) 
    
plt.figure()
plt.plot(tree)

del gcf2
del diff2
del diff_nozero
del gcf_level1
del gcf_level
del level1
del level2
del mask 
del tree
del path 
del areas

different = 0
path = path1[different]
areas = areas1[different]

nasa = imageio.imread(path + areas + "_NASADEM.tif")
xsar = imageio.imread(path + areas + "_XSAR.tif")
mask = np.invert(xsar == 0)

diff = (xsar - nasa)*mask
MM = diff.shape
del nasa
del xsar 

gcf = imageio.imread(path + areas + "_GCF_2000.tif")
gcf2 = resize(gcf, output_shape = MM, order = 1, mode = 'constant')
del gcf 
diff2 = np.reshape(diff, (1,MM[0]*MM[1]))

tree = []
for n in range(10): 
    level1 = gcf2 > n*10
    level2 = gcf2 < (n+1)*10
    gcf_level = level1*level2*mask
    gcf_level1 = np.reshape(gcf_level, (1,MM[0]*MM[1]))
    g_ind = np.nonzero(gcf_level1)
    diff_nozero = [diff2[0,k] for k in (list(g_ind[1]))]
    tree.append(np.sum(diff_nozero)/np.sum(gcf_level1)) 
    
plt.figure()
plt.plot(tree)

del gcf2
del diff2
del diff_nozero
del gcf_level1
del gcf_level
del level1
del level2
del mask 
del tree

different = 0
path = path1[different]
areas = areas1[different]

nasa = imageio.imread(path + areas + "_NASADEM.tif")
tdx = imageio.imread(path + areas + "_TDX.tif")
diff = (tdx - nasa)
MM = diff.shape

del nasa
del tdx 

glad = imageio.imread(path + areas + "_GLAD_2017_2000.tif")
glad2 = resize(glad, output_shape = MM, order = 1, mode = 'constant')
del glad 
mask = glad2 == 0
del glad2
gcf = imageio.imread(path + areas + "_GCF_2000.tif")
gcf2 = resize(gcf, output_shape = MM, order = 1, mode = 'constant')
del gcf 
diff2 = np.reshape(diff, (1,MM[0]*MM[1]))

tree = []
for n in range(10): 
    level1 = gcf2 > n*10
    level2 = gcf2 < (n+1)*10
    gcf_level = level1*level2*mask
    gcf_level1 = np.reshape(gcf_level, (1,MM[0]*MM[1]))
    g_ind = np.nonzero(gcf_level1)
    diff_nozero = [diff2[0,k] for k in (list(g_ind[1]))]
    tree.append(np.sum(diff_nozero)/np.sum(gcf_level1)) 
    
plt.figure()
plt.plot(tree)

del gcf2
del diff2
del diff_nozero
del gcf_level1
del gcf_level
del level1
del level2
del mask 
del tree


different = 1
path = path1[different]
areas = areas1[different]

nasa = imageio.imread(path + areas + "_NASADEM.tif")
tdx = imageio.imread(path + areas + "_TDX.tif")
diff = (tdx - nasa)
MM = diff.shape

del nasa
del tdx 

glad = imageio.imread(path + areas + "_GLAD_2017_2000.tif")
glad2 = resize(glad, output_shape = MM, order = 1, mode = 'constant')
del glad 
mask = glad2 == 0
del glad2
gcf = imageio.imread(path + areas + "_GCF_2000.tif")
gcf2 = resize(gcf, output_shape = MM, order = 1, mode = 'constant')
del gcf 
diff2 = np.reshape(diff, (1,MM[0]*MM[1]))

tree = []
for n in range(10): 
    level1 = gcf2 > n*10
    level2 = gcf2 < (n+1)*10
    gcf_level = level1*level2*mask
    gcf_level1 = np.reshape(gcf_level, (1,MM[0]*MM[1]))
    g_ind = np.nonzero(gcf_level1)
    diff_nozero = [diff2[0,k] for k in (list(g_ind[1]))]
    tree.append(np.sum(diff_nozero)/np.sum(gcf_level1)) 
    
plt.figure()
plt.plot(tree)
