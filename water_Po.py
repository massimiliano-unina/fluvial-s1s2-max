# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:02:43 2020

@author: massi
"""
import imageio 
from matplotlib import pyplot as plt
import numpy as np 
from tifffile import imsave 
import random
import os 
from scipy import ndimage, misc

#############
folder = r"C:\Users\massi\Downloads\\"
dir_list = os.listdir(folder)
dir_list.sort()

N_1 = {"139": 3, "168": 4}


print(dir_list)
N = 2
Out = 1
num = 1

ps1 = 128
r1 = 128


ps = 32
r = 8

enlarge = 64


rotation = [0,45,90, 135, 180, 225, 270, 315]
rotation1 = [0,45,90]
rotation2 = [0] #[0, 135]

A = len(rotation)
B = len(rotation1)
C = len(rotation2)


patches_iniziali = 0 
patches_finali = 0
gamma_0_file = folder + "Sigma0_VV.tif"
rhoLT_file = folder + "Sigma0_VH.tif"
output_file = folder + "feature_2.tif"
gamma_0 = imageio.imread(gamma_0_file)
rhoLT = imageio.imread(rhoLT_file)
output = imageio.imread(output_file)

gamma_0 = np.asarray(gamma_0)
rhoLT = np.asarray(rhoLT)
output = np.asarray(output)
output = output == 1
[s1, s2] = rhoLT.shape
x_train = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
y_train = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
x_val = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
y_val = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
x_test = np.ndarray(shape=(1, s1, s2, N), dtype='float32')
y_test = np.ndarray(shape=(1, s1,s2, Out), dtype='float32')
p2 = []
print(len(p2))
for y in range(1,s1-ps+1,r): 
    for x in range(1,s2-ps+1,r):
        s_0 = 0
        if s_0 == 0:
            p2.append([y,x])
p_train = []
p_val = []

P1 = len(p2)
p = p2#[p2[s] for s in v]  
random.shuffle(p)
p_train,p_val= p[:int(0.9*P1)],p[int(0.9*P1):P1]
print(len(p_train))
print(len(p_val))

P = len(p_train)
patches_finali += P
x_train_k = np.ndarray(shape=(P, ps, ps, N), dtype='float32')
y_train_k = np.ndarray(shape=(P, ps, ps, Out), dtype='float32')

n = 0
for patch in p_train:
    y0, x0 = patch[0], patch[1]
    x_train_k[n,:,:,0] = gamma_0[y0:y0+ps,x0:x0+ps]
    x_train_k[n,:,:,1] = rhoLT[y0:y0+ps,x0:x0+ps]
    y_train_k[n,:,:,0]= output[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#    y_train_k[n,:,:,1] = np.invert(output[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    n = n + 1
x_train = np.concatenate((x_train, x_train_k))
del x_train_k

y_train = np.concatenate((y_train, y_train_k))
del y_train_k

x_val_k = np.ndarray(shape=(len(p_val), ps, ps, N), dtype='float32')
y_val_k = np.ndarray(shape=(len(p_val), ps, ps, Out), dtype='float32')


n = 0
for patch in p_val:
    y0, x0 = patch[0], patch[1]
    x_val_k[n,:,:,0] = gamma_0[y0:y0+ps,x0:x0+ps]
    x_val_k[n,:,:,1] = rhoLT[y0:y0+ps,x0:x0+ps]
    y_val_k[n,:,:,0] = output[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#    y_val_k[n,:,:,1] = np.invert(output[y0:y0+ps, x0:x0+ps])#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]

    n = n + 1
x_val = np.concatenate((x_val, x_val_k))
del x_val_k

y_val = np.concatenate((y_val, y_val_k))
del y_val_k
num +=1
#p2 = []
#print(len(p2))
#for y in range(1,s1-ps1+1,r1): 
#    for x in range(1,s2-ps1+1,r1):
#        if s_0 == 0:
#            p2.append([y,x,materials])
#p_test = p2
#
#P1 = len(p_test)
#
#P = len(p_test)
#patches_finali += P
#x_test_k = np.ndarray(shape=(P, ps1, ps1, N), dtype='float32')
#y_test_k = np.ndarray(shape=(P, ps1, ps1, Out), dtype='float32')
#
#n = 0
#for patch in p_test:
#    y0, x0 = patch[0], patch[1]
#    
##                y_train_k[n,:,:,3] = corine_INVALID[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#    n = n + 1
#x_test = np.concatenate((x_test, x_test_k))
#del x_test_k
#
#y_test = np.concatenate((y_test, y_test_k))
#del y_test_k
#
#num +=1
#

x_test[0,:,:,0] = gamma_0
x_test[0,:,:,1] = rhoLT

y_test[0,:,:,0]= output
#y_test[0,:,:,1] = np.invert(output)#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]

np.savez("train_data_water_Po.npz",x_test = x_test, y_test= y_test, x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val)    
