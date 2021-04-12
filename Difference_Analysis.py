# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:39:34 2019

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

N_Images = 5
mean_diff = np.zeros(255)
X4 = []
TX4 = []
N4 = []
LAND4 = []
DIFF4 = []
DIFFXTX = []
una_tra_tante = [2]
piu_tra_tante = [0,1,3,4,5]
for kkk1 in piu_tra_tante: #  range(N_Images): #
    
    imgs2 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_SRTM_daruotare"+str(kkk1 +1)+"_fatto.tif")
    imgx2 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_XSAR_daruotare"+str(kkk1 +1)+"_fatto.tif")
    imgn2 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_NASADEM_daruotare"+str(kkk1 +1)+"_fatto.tif")
    imgtx2 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_TDX_daruotare"+str(kkk1 +1)+"_fatto.tif")
    land = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_landcover_daruotare"+str(kkk1 +1)+"_fatto.tif")
#    imgs2 = imgs2*(imgs2>=0) + np.min(imgs2)*( 1 - imgs2>=0) 
#    imgx2 = imgx2*(imgx2>=0) + np.min(imgx2)*( 1 - imgx2>=0)
#    imgtx2 = imgtx2*(imgtx2>=0)
#    tx = np.reshape(imgtx2*(imgtx2 > 0), (imgtx2.shape[0]*imgtx2.shape[1]))
#    imgtx2_m = (np.nonzero(tx))
#    iii = [tx[kk] for kk in imgtx2_m]
#    
#    imgtx2_min = np.min(iii)#imgtx2_m[0]
#    print(imgtx2_min)
#    imgtx2 = imgtx2*(imgtx2>=0) + imgtx2_min*( 1 - imgtx2>=0)
#    imgn2 = imgn2*(imgn2>=0) + np.min(imgn2)*( 1 - imgn2>=0)
    
#    imgs3 = imgs2 - np.mean(imgs2)
#    imgn3 = imgn2 - np.mean(imgn2)
#    imgtx3 = imgtx2 - np.mean(imgtx2)
#    imgx3 = imgx2 - np.mean(imgx2)
#
#    slopex3 = sobel_gradientM(imgx3)   
#    slopetx3 = sobel_gradientM(imgtx3)   

    
    diff22 = np.pad(imgx2,((1,0),(0,0)),'constant') - np.pad(imgx2,((0,1),(0,0)),'constant')
    
    plt.figure()
    plt.imshow(diff22, cmap='gray')
    ndvi3 = np.asarray(diff22)#imgx - fimg3_T)
    im3 = r"D:\Works\DLR\Uganda_Solberg\\Diff_"+str(kkk1 +1)+".tif"
    imsave(im3, ndvi3)

    imgtx4 = np.reshape(imgtx2, (imgtx2.shape[0]*imgtx2.shape[1]))

    imgx4 = np.reshape(imgx2, (imgx2.shape[0]*imgx2.shape[1]))
    imgn4 = np.reshape(imgn2, (imgx2.shape[0]*imgx2.shape[1]))
    diff = imgx4 - imgtx4
    diffxn = imgx2 - imgn2
    diffxtx = imgx2 - imgtx2
    difftx = imgx4 - imgtx4
    land4 = np.reshape(land, (land.shape[0]*land.shape[1]))
    mean_tree = np.zeros(shape=(land.shape[0],land.shape[1]))
    tree_level1 = []
    for level in range(255):
        land_level = land == level 
        
        diff_level = diffxn*land_level
        tree_level = (np.sum(diff_level))/(np.count_nonzero(land_level) + 10**(-10))
        tree_level1.append((np.sum(diff_level))/(np.count_nonzero(land_level) + 10**(-10)))
        mean_tree += tree_level*land_level
    

#    imgx5 = imgx2 - tree_level1[0]    
#    imgx5 = imgx2 - mean_tree 
    imgx4 = np.reshape(imgx2, (imgx2.shape[0]*imgx2.shape[1]))
    diffxn2 = np.reshape(diffxn, (imgx2.shape[0]*imgx2.shape[1]))
    for n in range(0,len(imgx4),100):
        X4.append(imgx4[n] + 10**(-10))
        TX4.append(imgtx4[n] + 10**(-10))
        N4.append(imgn4[n] + 10**(-10))
        DIFF4.append(diffxn2[n])
        DIFFXTX.append(difftx[n])
        LAND4.append(land4[n])

#print(LAND4)
diff_mean = np.mean(DIFF4)
plt.figure()
plt.scatter(LAND4, DIFF4)
plt.title("Penetration bias Band X-C vs Tree Cover with Difference in DEM = " + str(diff_mean) )
plt.xlabel("Tree Cover")
plt.ylabel("X-C")

plt.figure()
plt.scatter(LAND4, DIFFXTX)
plt.title("Penetration bias Band X-TX vs Tree Cover ")
plt.xlabel("Tree Cover")
plt.ylabel("X-TX")

plt.figure()
plt.scatter(X4, TX4)
plt.title("Band TX vs Band X ")
plt.xlabel("X")
plt.ylabel("TX")

plt.figure()
plt.scatter(X4, N4)
plt.title("Band C vs Band X ")
plt.xlabel("X")
plt.ylabel("C")

#n2 = N4
##tx2 = np.reshape(imgtx2, (imgn2.shape[0]*imgn2.shape[1]))
#x2 = X4
A = np.vstack([N4, np.ones(len(N4))]).T
m,c = np.linalg.lstsq(A, X4, rcond=None)[0]



for kkk1 in una_tra_tante: #  range(N_Images): #
    
#    m = {}
#    c = {}
#    for level in range(255): 
#        land_level = LAND4 == level*np.ones(len(LAND4)) 
#        if np.count_nonzero(land_level) == 0: 
#            m[level], c[level] = 1,0
#        else: 
#            n1 = np.nonzero(N4*land_level)
#            n2 = [N4[indi] for indi in n1[0]]
#            x1 = np.nonzero(X4*land_level)
#            x2 = [X4[indi] for indi in x1[0]]
#            A = np.vstack([n2, np.ones(len(n2))]).T
#            m[level], c[level] = np.linalg.lstsq(A, x2, rcond=None)[0]
    
    imgn2 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_NASADEM_daruotare"+str(kkk1 +1)+"_fatto.tif")
    imgx2 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_XSAR_daruotare"+str(kkk1 +1)+"_fatto.tif")
    imgtx2 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_TDX_daruotare"+str(kkk1 +1)+"_fatto.tif")
    
    land = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_landcover_daruotare"+str(kkk1 +1)+"_fatto.tif")
    X_est = np.zeros(imgn2.shape)
    
    n2 = np.reshape(imgn2, (imgn2.shape[0]*imgn2.shape[1]))
    tx2 = np.reshape(imgtx2, (imgn2.shape[0]*imgn2.shape[1]))
    x2 = np.reshape(imgx2, (imgx2.shape[0]*imgn2.shape[1]))
    A1 = np.vstack([n2, np.ones(len(n2))]).T
    m2,c2 = np.linalg.lstsq(A1, x2, rcond=None)[0]
    m1 = np.asarray(m2, dtype=np.float32)
    c1 = np.asarray(c2, dtype=np.float32)
    X_est = np.zeros(imgn2.shape)
    X_est = m1*imgn2 + c1    
    X_est2 = m*imgn2 + c   
    diff = imgx2 - imgn2
    mean_tree2 = np.zeros(shape=(land.shape[0],land.shape[1]))
    mean_tree2b = np.zeros(shape=(land.shape[0],land.shape[1]))
    tree_level1b = []
    for level in range(255):
        land_level = land == level 
        tree_level2 = tree_level1[level]
        mean_tree2 += tree_level2*land_level

        diff_levelb = diff*land_level
        tree_levelb = (np.sum(diff_levelb))/(np.count_nonzero(land_level) + 10**(-10))
        tree_level1b.append((np.sum(diff_levelb))/(np.count_nonzero(land_level) + 10**(-10)))
        mean_tree2b += tree_levelb*land_level
        
        
    ndvi3 = np.asarray(mean_tree2)#imgx - fimg3_T)
    im3 = r"D:\Works\DLR\Uganda_Solberg\uganda_TreeDiffPre_daruotare"+str(kkk1 +1)+"_estimation.tif"
    imsave(im3, ndvi3)
    
    
    ndvi3 = np.asarray(mean_tree2b)#imgx - fimg3_T)
    im3 = r"D:\Works\DLR\Uganda_Solberg\uganda_TreeDiffPost_daruotare"+str(kkk1 +1)+"_estimation.tif"
    imsave(im3, ndvi3)#    for level in range(255): 
#        land_level1 = land == level
#        m1 = np.asarray(m[level], dtype=np.float32)
#        c1 = np.asarray(c[level], dtype=np.float32)
#        print("il coeff m1 per tree cover pari a " + str(level) + " : " + str(m1))
#        print("il coeff c1  per tree cover pari a " + str(level) + " : " + str(c1))
#        X_est = X_est + (m1*imgn2 + c1)*land_level1 
#
#    ndvi3 = np.asarray(X_est)#imgx - fimg3_T)
#    im3 = r"D:\Works\DLR\Uganda_Solberg\2uganda_NASADEM_daruotare"+str(kkk1 +1)+"_estimation.tif"
#    imsave(im3, ndvi3)#    for level in range(255): 


##    print(slopex3.shape)
##    plt.figure()
##    plt.subplot(211)
##    plt.imshow(imgx3, cmap='gray')
##    plt.subplot(212)
##    plt.imshow(slopex3, cmap='gray')
##    
##    plt.figure()
##    plt.subplot(211)
##    plt.imshow(imgtx3, cmap='gray')
##    plt.subplot(212)
##    plt.imshow(slopetx4, cmap='gray')
#    
#
#    imgs2 = gaussian_filter(imgs2, (3,3))
#    mean_s = np.mean(imgs2)
#    mean_n = np.mean(imgn2)
##    imgn2 = gaussian_filter(imgn2, (4,4))
#    imgx2 = gaussian_filter(imgx2, (3,3))
#    diffxtx = imgtx2 - imgx2 
#    diff2 = imgn2 - imgtx2
#    diff = imgn2 - imgs2 
#    imgn2 = imgn2 - diff #- mean_n + mean_s #
##    imgx2 = imgx2 - diff
##    diff3 = imgn2 - imgtx2
#    diff_mean2 = np.mean(diff2, axis=0)
##    diff_mean3 = np.mean(diff3, axis=0)
#    land_mean = np.mean(land, axis = 0)
##    plt.figure()
##    plt.subplot(311)
##    plt.plot(diff_mean2)
##    plt.subplot(312)
##    plt.plot(diff_mean3)
##    plt.subplot(313)
##    plt.plot(land_mean)
#
#    ndvi3 = np.asarray(imgn2)#imgx - fimg3_T)
#    im3 = r"D:\Works\DLR\Uganda_Solberg\\NASADEM_corrected_"+str(kkk1 +1)+".tif"
#    imsave(im3, ndvi3)
#
##    ndvi3 = np.asarray(imgx2)#imgx - fimg3_T)
##    im3 = r"D:\Works\DLR\Uganda_Solberg\\XSAR_corrected_"+str(kkk1 +1)+".tif"
##    imsave(im3, ndvi3)
#
#
#        
##    for level1 in range(20): 
##        land_level_D = land > level1*255/20
##        land_level_U = land < (level1 + 1)*255/20
##        land_level = land_level_D*land_level_U
##        x = imgx2*land_level
##        x2 = np.reshape(x, (1,imgx2.shape[0]*imgx2.shape[1]))
##        x3 = np.nonzero(x2)
##        x_mean = np.median(x3)
##        tx = imgtx2*land_level
##        tx2 = np.reshape(tx, (1,imgx2.shape[0]*imgx2.shape[1]))
##        tx3 = np.nonzero(tx2)
##        tx_mean = np.median(tx3)
##        imgx2 = imgx2 - x_mean*land_level 
##        imgtx2 = imgtx2 - tx_mean*land_level 
##        
#        
##    im = imgs3 - imgn3 
#    im = imgs2 - imgn2 
#    imxtx = imgx3 - imgtx3
##    imxtx = imgx2 - imgtx2
#    
#    diff_level = []
#    diff_level2 = []
#    [N,M] = imxtx.shape
##    diff_m = np.mean(imxtx, axis=0)
###    plt.figure()
###    plt.plot(diff_m)
##    fimg2_T = np.ndarray(shape=(0,M), dtype='float32')
##    fimg21 = np.reshape(diff_m, newshape = (1,M))
##    
##    for k in range(N):
##        fimg2_T = np.concatenate((fimg2_T, fimg21))
#    
##    median_diff_im = np.zeros(imxtx.shape)
##    for level in range(20):
##        print(str(level*255/20))
##        print(str((level + 1)*255/20))
##        land_level_D = land > level*255/20
##        land_level_U = land < (level + 1)*255/20
##        land_level = land_level_D*land_level_U
##        
#    imxtx2 = imxtx
#    ndvi3 = np.asarray(imxtx2)#imgx - fimg3_T)
#    im3 = r"D:\Works\DLR\Uganda_Solberg\\PreNoise_"+str(kkk1 +1)+".tif"
#    imsave(im3, ndvi3)
##    
#    for level in range(255):
#        land_level = land == level 
##    for level in range(20):
##        print(str(level*255/20))
##        print(str((level + 1)*255/20))
##        land_level_D = land > level*255/20
##        land_level_U = land < (level + 1)*255/20
##        land_level = land_level_D*land_level_U
#        
#        diff = imxtx*land_level
#        diff2 = np.reshape(diff, (diff.shape[0]*diff.shape[1]))
#        diff2 = np.asarray(diff2)
##        print(diff2.shape)
#        NON = np.nonzero(diff2)
##        print(print(NON))
#        diff3 = [diff2[kkkk] for kkkk in NON]
#        diff3 = np.asarray(diff3)
##        M = diff3.shape[1]
###        print(M)
##        if M == 0: 
##            median_diff = 0
##        else: 
##            median_diff = diff3[0,M//2]
#        mean_diff[level] = np.sum(diff3)/(np.count_nonzero(land_level) + 10**(-10)) 
##        
##        if np.count_nonzero(land_level) == 0: 
##            median_diff = np.sum(diff3)/(np.count_nonzero(land_level) + 10**(-10))
###            mean_diff = np.sum(diff)/(np.count_nonzero(land_level) + 10**(-10))
###        diff_level.append(median_diff)
##        diff_level2.append(mean_diff)
##        median_diff_im = median_diff_im + mean_diff*land_level 
##        imxtx2 = imxtx2 - mean_diff*land_level 
##        imgx2 = imgx2 - (median_diff + fimg2_T)*land_level 
#
##    diff_m = np.mean(imxtx2, axis=0)
###    plt.figure()
###    plt.plot(diff_m)
##
##    fimg2_T = func_1d_2_column2d(diff_m, imxtx2)
##    print(type(fimg2_T))
###    X = imgx2 - fimg2_T + median_diff_im
##    
###    plt.figure()
###    plt.subplot(211)
###    plt.plot(diff_level)
###    plt.subplot(212)
###    plt.plot(diff_level2)
##
##    
##    magnitude_spectrum1, fshift1 = compute_fft(fimg2_T)
##    print(np.mean(magnitude_spectrum1))
##    print(np.min(magnitude_spectrum1))
##    print(np.max(magnitude_spectrum1))
###    magnitude_spectrum2 = magnitude_spectrum1 > (np.max(magnitude_spectrum1) - np.min(magnitude_spectrum1))*0.75
####    struct2 = generate_binary_structure(3,3)
###    magnitude_spectrum2 = morphology.binary_dilation(magnitude_spectrum2).astype(magnitude_spectrum2.dtype)
###    ,structure=struct2
##    
##    plt.figure()
##    plt.imshow(magnitude_spectrum1,cmap='gray')
###    plt.subplot(212)
###    plt.imshow(magnitude_spectrum2,cmap='gray')
###    
##    
###    plt.figure()
###    plt.imshow(imtx2,cmap='gray')
##    
###    plt.figure()
###    plt.subplot(211)
###    plt.plot(np.mean(imtx2,axis=0))
###    plt.subplot(212)
###    plt.plot(np.mean(land,axis=0))
#    plt.figure()
#    plt.plot(mean_diff)
#
##plt.figure()
##plt.plot(mean_diff/N_Images)
##
##for kkk1 in range(N_Images):
##    land1 = imageio.imread(r"D:\Works\DLR\Uganda_Solberg\uganda_landcover_daruotare"+str(kkk1 +1)+"_fatto.tif")
##    
##    median_diff_im = np.zeros(land1.shape)
##    for level in range(255):
##        land_level1 = land1 == level 
##        median_diff_im = median_diff_im + (mean_diff[level]/N_Images)*land_level1 
##    
##    
##    ndvi3 = np.asarray(median_diff_im)#imgx - fimg3_T)
##    im3 = r"D:\Works\DLR\Uganda_Solberg\\Mean_fortreetype_"+str(kkk1 +1)+".tif"
##    imsave(im3, ndvi3)
##
##    ndvi3 = np.asarray(fimg2_T)#imgx - fimg3_T)
##    im3 = r"D:\Works\DLR\Uganda_Solberg\\Noise_"+str(kkk1 +1)+".tif"
##    imsave(im3, ndvi3)
##
##
###    IM = imgx2 - imgn2
###    IMM = np.mean(IM, axis = 0)
###    IMtx = imgx2 - imgtx2
####    IMtx = IMtx*(IMtx < 30000)
###    IMtxM = np.mean(IMtx, axis = 0)
###    Land = np.mean(land, axis = 0)
###    IMx = imgtx2
###    IMxM = np.mean(IMx, axis = 0)
####    plt.figure()
####    plt.subplot(211)
####    plt.plot(IMtxM)#, cmap='gray')
####    plt.subplot(212)
####    plt.plot(Land)#, cmap='gray')
###    
###    values_tx = []
###    values_x = []
###    for level in range(255): 
###        land_level = land == level 
###        IMtx2 = IMtx*land_level
###        IMx2 = imgx2*land_level
####        print(level)
####        print(np.count_nonzero(land_level))
###        values_tx.append(np.sum(IMtx2)/(np.count_nonzero(land_level) + 10**(-10)))
####        values_x.append(np.sum(IMx2)/(np.count_nonzero(land_level) + 10**(-10)))
###    
###    
###    plt.figure()
###    plt.plot(values_tx, 'r')
####    plt.plot(values_x,'b')
###    IM2 = np.reshape(IMtx,(1,IM.shape[0]*IM.shape[1]))
###    land2 = np.reshape(land,(1,land.shape[0]*land.shape[1]))
####    plt.figure()
####    plt.scatter(land2,IM2)