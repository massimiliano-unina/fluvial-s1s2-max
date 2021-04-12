# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:20:23 2019

@author: massi
"""

from __future__ import print_function

import sys
import os
from tifffile import imsave
import numpy as np
import gdal
import imageio
from scipy.misc import imresize
from scipy.ndimage import gaussian_filter, morphology,generate_binary_structure


def test_data(dataset_folder, patch_side,patch_side_2,indices,indices1):
    
    #############
    # short names
    path, ps, ps2,v,v1 = dataset_folder, patch_side,patch_side_2, indices,indices1
    #############
    dir_list = os.listdir(path)
    dir_list.sort()
    print(dir_list)
    N = 5
    Out = 3
    import random
    num = 1
    x_train = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
    y_train = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
    vh_file =  '1_Albufera_VH2.tif'
    dataset = gdal.Open(path + vh_file, gdal.GA_ReadOnly)
    vh1 = dataset.ReadAsArray()
    dataset = None
    print(vh1.shape)
    x_val = np.ndarray(shape=(0, vh1.shape[1], vh1.shape[2], N), dtype='float32')
    y_val = np.ndarray(shape=(0, vh1.shape[1], vh1.shape[2], Out), dtype='float32')
    
    x_train2 = np.ndarray(shape=(0, ps2, ps2, N), dtype='float32')
    y_train2 = np.ndarray(shape=(0, ps2,ps2, Out), dtype='float32')
    x_val2 = np.ndarray(shape=(0, ps2, ps2, N), dtype='float32')
    y_val2 = np.ndarray(shape=(0, ps2,ps2, Out), dtype='float32')
    
    r = 16
    for file in dir_list:
        print(file)
        if file.lower().find('albufera_vv.tif') != -1 and file[0]==str(num): #and file[2]==str(date[num]) and date[num] < 10: #and file[2]<str(7):
            vv_file = file
            print(vv_file)
            park_file = 'park' + vv_file[-4:]
            veg_file =  str(num) + '_vegetation' + vv_file[-4:]
            wat_file =  str(num) + '_water' + vv_file[-4:]
            soil_file =  str(num) + '_bare_soil' + vv_file[-4:]
#            soil_file_2 =  str(num) + '_cloud_low_proba' + vv_file[-4:]
            
            dataset = gdal.Open(path + veg_file, gdal.GA_ReadOnly)
            veg = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path + wat_file, gdal.GA_ReadOnly)
            wat = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path + soil_file, gdal.GA_ReadOnly)
            soil = dataset.ReadAsArray()
            dataset = None
#            dataset = gdal.Open(path + soil_file_2, gdal.GA_ReadOnly)
#            soil_2 = dataset.ReadAsArray()
#            dataset = None
#            soil = soil + soil_2
            
            vh_file =  str(num) + '_Albufera_VH' + vv_file[-4:]
            dataset = gdal.Open(path + vh_file, gdal.GA_ReadOnly)
            vh = dataset.ReadAsArray()
            dataset = None

            dataset = gdal.Open(path + vv_file, gdal.GA_ReadOnly)
            vv = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path + park_file, gdal.GA_ReadOnly)
            park = dataset.ReadAsArray()
            dataset = None
            veg = veg[0,:,:]/255
            soil = soil[0,:,:]/255
            wat = wat[0,:,:]/255
            park = 1 - park/255
            vv = vv[0,:,:]
            vh = vh[0,:,:]
#### other additional input    
            [s1, s2] = vv.shape
            vv_vh = (vv + 10**(-10))/(vh + 10**(-10))  #vv_vh[0,:,:]
            ave_vv_vh =  (vv + vh)/2 #np.zeros(shape=(s1,s2))#ave_vv_vh[0,:,:]
            diff_vv_vh = vv - vh #diff_vv_vh[0,:,:]
            p2 = []
            print(len(p2))
            for y in range(1,s1-ps+1,r): 
                for x in range(1,s2-ps+1,r):
                    mask_d0 = park[y:y+ps,x:x+ps]
                    mask_veg = veg[y:y+ps,x:x+ps]
                    mask_wat = wat[y:y+ps,x:x+ps]
                    mask_soil = soil[y:y+ps,x:x+ps]
                    [m1,m2] = mask_d0.shape
                    s_0 =  mask_d0.sum()
                    mask_T= mask_veg + mask_wat + mask_soil
                    s_T =  mask_T.sum()
                    if s_0 == 0 and s_T > (0.9)*(m1*m2):
                        p2.append([y,x])
                        
            print(len(p2))
            if len(p2) < 700: 
                
                p =[v1[str(s)] for s in range(len(p2))]  # p2#[v[s] for s in range(len(p2))] 
            else: 
                p =[v[str(s)] for s in range(len(p2))]  # p2#[v[s] for s in range(len(p2))] 
#            random.shuffle(p)
            P = int(2000)
            p_train,p_val= p[:int(0.8*P)],p[int(0.8*P):P]
            print(len(p_train))
            print(len(p_val))

#            x_train_k = np.ndarray(shape=(len(p_train), ps, ps, N), dtype='float32')
#            y_train_k = np.ndarray(shape=(len(p_train), ps, ps, Out), dtype='float32')
#            x_train_k2 = np.ndarray(shape=(len(p_train), ps2, ps2, N), dtype='float32')
#            y_train_k2 = np.ndarray(shape=(len(p_train), ps2, ps2, Out), dtype='float32')
#            n = 0
#            for patch in p_train:
#                y0, x0 = patch[0], patch[1]
##                x_train_k[n,:,:,0] = vv[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,:,:,1] = vh[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,:,:,2] = vv_vh[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,:,:,3] = ave_vv_vh[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,:,:,4] = diff_vv_vh[y0:y0+ps,x0:x0+ps]
###                tren[y0:y0+ps,x0:x0+ps] += 1
##                
##                y_train_k[n,:,:,0]= soil[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
##                y_train_k[n,:,:,1] = veg[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
##                y_train_k[n,:,:,2] = wat[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
##                
##                x_train_k2[n,:,:,0] = vv[y0:y0+ps2,x0:x0+ps2]
##                x_train_k2[n,:,:,1] = vh[y0:y0+ps2,x0:x0+ps2]
##                x_train_k2[n,:,:,2] = vv_vh[y0:y0+ps2,x0:x0+ps2]
##                x_train_k2[n,:,:,3] = ave_vv_vh[y0:y0+ps2,x0:x0+ps2]
##                x_train_k2[n,:,:,4] = diff_vv_vh[y0:y0+ps2,x0:x0+ps2]
###                tren[y0:y0+ps,x0:x0+ps] += 1
##                
##                y_train_k2[n,:,:,0]= soil[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
##                y_train_k2[n,:,:,1] = veg[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
##                y_train_k2[n,:,:,2] = wat[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                n = n + 1
#            x_train = np.concatenate((x_train, x_train_k))
#            y_train = np.concatenate((y_train, y_train_k))
#            x_train2 = np.concatenate((x_train2, x_train_k2))
#            y_train2 = np.concatenate((y_train2, y_train_k2))


            x_val_k = np.ndarray(shape=(1, vh1.shape[1], vh1.shape[2], N), dtype='float32')
            y_val_k = np.ndarray(shape=(1, vh1.shape[1], vh1.shape[2], Out), dtype='float32')
            
            n = 0
#            for patch in p_val:
            x_val_k[n,:,:,0] = vv
            x_val_k[n,:,:,1] = vh
            x_val_k[n,:,:,2] = vv_vh
            x_val_k[n,:,:,3] = ave_vv_vh
            x_val_k[n,:,:,4] = diff_vv_vh
#                vald[y0:y0+ps,x0:x0+ps] += 1
            y_val_k[n,:,:,0] = soil#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_val_k[n,:,:,1] = veg#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            y_val_k[n,:,:,2] = wat#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            
                
#                n = n + 1
            x_val = np.concatenate((x_val, x_val_k))
            y_val = np.concatenate((y_val, y_val_k))
            vv, park, veg, wat, soil= None, None,None,None,None 
            num +=1

    return x_train, y_train, x_val, y_val, x_train2, y_train2, x_val2, y_val2#, tren, vald,




def load_data(dataset_folder, patch_side,patch_side_2,indices,indices1):
    
    #############
    # short names
    path, ps, ps2,v,v1 = dataset_folder, patch_side,patch_side_2, indices,indices1
    #############
    dir_list = os.listdir(path)
    dir_list.sort()
    print(dir_list)
    N = 5
    Out = 3
    import random
    num = 1
    x_train = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
    y_train = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
    x_val = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
    y_val = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
    
    x_train2 = np.ndarray(shape=(0, ps2, ps2, N), dtype='float32')
    y_train2 = np.ndarray(shape=(0, ps2,ps2, Out), dtype='float32')
    x_val2 = np.ndarray(shape=(0, ps2, ps2, N), dtype='float32')
    y_val2 = np.ndarray(shape=(0, ps2,ps2, Out), dtype='float32')
    
#    tren = np.zeros(shape=(2800,2700),dtype='float32')
#    vald = np.zeros(shape=(2800,2700),dtype='float32')
    r = 16
    for file in dir_list:
        print(file)
        if file.lower().find('albufera_vv.tif') != -1 and file[0]==str(num):# and num < 2:
            vv_file = file
            print(vv_file)
            park_file ='park' + vv_file[-4:] #str(num) + '_
            veg_file =  str(num) + '_vegetation' + vv_file[-4:]
            wat_file =  str(num) + '_water' + vv_file[-4:]
            soil_file =  str(num) + '_bare_soil' + vv_file[-4:]
            patch_file =  'Patches_0' + vv_file[-4:]
            dataset = gdal.Open(path + patch_file, gdal.GA_ReadOnly)
            pat0 = dataset.ReadAsArray()
            dataset = None
            patch_file =  'Patches_1' + vv_file[-4:]
            dataset = gdal.Open(path + patch_file, gdal.GA_ReadOnly)
            pat1 = dataset.ReadAsArray()
            dataset = None
            patch_file =  'Patches_2' + vv_file[-4:]
            dataset = gdal.Open(path + patch_file, gdal.GA_ReadOnly)
            pat2 = dataset.ReadAsArray()
            dataset = None
            patch_file =  'Patches_3' + vv_file[-4:]
            dataset = gdal.Open(path + patch_file, gdal.GA_ReadOnly)
            pat3 = dataset.ReadAsArray()
            dataset = None
            patch_file =  'Patches_4' + vv_file[-4:]
            dataset = gdal.Open(path + patch_file, gdal.GA_ReadOnly)
            pat4 = dataset.ReadAsArray()
            dataset = None
#            soil_file_2 =  str(num) + '_cloud_low_proba' + vv_file[-4:]
            
            dataset = gdal.Open(path + veg_file, gdal.GA_ReadOnly)
            veg = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path + wat_file, gdal.GA_ReadOnly)
            wat = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path + soil_file, gdal.GA_ReadOnly)
            soil = dataset.ReadAsArray()
            dataset = None
#            dataset = gdal.Open(path + soil_file_2, gdal.GA_ReadOnly)
#            soil_2 = dataset.ReadAsArray()
#            dataset = None
#            soil = soil + soil_2
            
            vh_file =  str(num) + '_Albufera_VH' + vv_file[-4:]
            dataset = gdal.Open(path + vh_file, gdal.GA_ReadOnly)
            vh = dataset.ReadAsArray()
            dataset = None

            dataset = gdal.Open(path + vv_file, gdal.GA_ReadOnly)
            vv = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path + park_file, gdal.GA_ReadOnly)
            park = dataset.ReadAsArray()
            dataset = None
            veg = veg[0,:,:]/255
            soil = soil[0,:,:]/255
            wat = wat[0,:,:]/255
            park = 1 - park/255
            pato = pat0*pat1*pat2*pat3*pat4
#            pat0 = 1-pat0
#            pat1 = 1-pat1
#            pat2 = 1-pat2
#            pat3 = 1-pat3
#            pat4 = 1-pat4
            park = park + (1 - pato)
            imsave(path + park_file[:-4] + "_quellousato.tif",park)
            vv = vv[0,:,:]
            vh = vh[0,:,:]
            back = 1 - veg*wat*soil
#### other additional input    
            [s1, s2] = vv.shape
            vv_vh = (vv + 10**(-10))/(vh + 10**(-10))  #vv_vh[0,:,:]
            ave_vv_vh =  (vv + vh)/2 #np.zeros(shape=(s1,s2))#ave_vv_vh[0,:,:]
            diff_vv_vh = vv - vh #diff_vv_vh[0,:,:]
            p2 = []
            print(len(p2))
            for y in range(1,s1-ps+1,r): 
                for x in range(1,s2-ps+1,r):
                    mask_d0 = park[y:y+ps,x:x+ps]
                    mask_veg = veg[y:y+ps,x:x+ps]
                    mask_wat = wat[y:y+ps,x:x+ps]
                    mask_soil = soil[y:y+ps,x:x+ps]
                    [m1,m2] = mask_d0.shape
                    s_0 =  mask_d0.sum()
                    mask_T= mask_veg + mask_wat + mask_soil
                    s_T =  mask_T.sum()
                    if s_0 == 0 and s_T > (0.75)*(m1*m2):
                        p2.append([y,x])
            p_train = []
            p_val = []
            print(len(p2))
            ### TEST
            adic = {}
            adic = v['dic']
            indi = adic['p']

            adic1 = {}
            adic1 = v1['dic']
            indi1 = adic1['p']
###############
#            if len(p2) > 800: 
#    
#                p = [p2[s] for s in range(0,len(p2),50)]#,int((len(p2) - 800)))]
#            else: 
#                p = [p2[s] for s in range(0,700,50)]
#            print(p)
#            p_val = p
            ######
#            p = [p2[s] for s in indi]
            ########
#            if num == 1:
#                p_val = p[:int(0.2*P)]
#                
#            else: 
#                p_train = p[:int(0.8*P)]
            
#############            
#            p =[p2[v[str(s)]] for s in range(700)]  # p2#[v[s] for s in range(len(p2))]                    
    ########              
            P = 3000#len(p2)#int(3000)
            p = p2#[p2[s] for s in v]  
            random.shuffle(p)
            p_train,p_val= p[:int(0.9*P)],p[int(0.9*P):P]
            print(len(p_train))
            print(len(p_val))

            x_train_k = np.ndarray(shape=(len(p_train), ps, ps, N), dtype='float32')
            y_train_k = np.ndarray(shape=(len(p_train), ps, ps, Out), dtype='float32')
#            x_train_k2 = np.ndarray(shape=(len(p_train), ps2, ps2, N), dtype='float32')
#            y_train_k2 = np.ndarray(shape=(len(p_train), ps2, ps2, Out), dtype='float32')
            n = 0
            for patch in p_train:
                y0, x0 = patch[0], patch[1]
                x_train_k[n,:,:,0] = vv[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,1] = vh[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,2] = vv_vh[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,3] = ave_vv_vh[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,4] = diff_vv_vh[y0:y0+ps,x0:x0+ps]
#                tren[y0:y0+ps,x0:x0+ps] += 1
                
                y_train_k[n,:,:,0]= soil[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_train_k[n,:,:,1] = veg[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_train_k[n,:,:,2] = wat[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_train_k[n,:,:,3] = back[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                x_train_k2[n,:,:,0] = vv[y0:y0+ps2,x0:x0+ps2]
#                x_train_k2[n,:,:,1] = vh[y0:y0+ps2,x0:x0+ps2]
#                x_train_k2[n,:,:,2] = vv_vh[y0:y0+ps2,x0:x0+ps2]
#                x_train_k2[n,:,:,3] = ave_vv_vh[y0:y0+ps2,x0:x0+ps2]
#                x_train_k2[n,:,:,4] = diff_vv_vh[y0:y0+ps2,x0:x0+ps2]
##                tren[y0:y0+ps,x0:x0+ps] += 1
#                
#                y_train_k2[n,:,:,0]= soil[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_train_k2[n,:,:,1] = veg[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_train_k2[n,:,:,2] = wat[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
##                y_train_k2[n,:,:,3] = back[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                n = n + 1
            x_train = np.concatenate((x_train, x_train_k))
            y_train = np.concatenate((y_train, y_train_k))
#            x_train2 = np.concatenate((x_train2, x_train_k2))
#            y_train2 = np.concatenate((y_train2, y_train_k2))


            x_val_k = np.ndarray(shape=(len(p_val), ps, ps, N), dtype='float32')
            y_val_k = np.ndarray(shape=(len(p_val), ps, ps, Out), dtype='float32')
#            x_val_k2 = np.ndarray(shape=(len(p_val), ps2, ps2, N), dtype='float32')
#            y_val_k2 = np.ndarray(shape=(len(p_val), ps2, ps2, Out), dtype='float32')
            
            n = 0
            for patch in p_val:
                y0, x0 = patch[0], patch[1]
                x_val_k[n,:,:,0] = vv[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,1] = vh[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,2] = vv_vh[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,3] = ave_vv_vh[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,4] = diff_vv_vh[y0:y0+ps,x0:x0+ps]
#                vald[y0:y0+ps,x0:x0+ps] += 1
                y_val_k[n,:,:,0] = soil[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_val_k[n,:,:,1] = veg[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_val_k[n,:,:,2] = wat[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_val_k[n,:,:,3] = back[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                x_val_k2[n,:,:,0] = vv[y0:y0+ps2,x0:x0+ps2]
#                x_val_k2[n,:,:,1] = vh[y0:y0+ps2,x0:x0+ps2]
#                x_val_k2[n,:,:,2] = vv_vh[y0:y0+ps2,x0:x0+ps2]
#                x_val_k2[n,:,:,3] = ave_vv_vh[y0:y0+ps2,x0:x0+ps2]
#                x_val_k2[n,:,:,4] = diff_vv_vh[y0:y0+ps2,x0:x0+ps2]
##                vald[y0:y0+ps,x0:x0+ps] += 1
#                y_val_k2[n,:,:,0] = soil[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_val_k2[n,:,:,1] = veg[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_val_k2[n,:,:,2] = wat[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
##                y_val_k2[n,:,:,3] = back[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                n = n + 1
            x_val = np.concatenate((x_val, x_val_k))
            y_val = np.concatenate((y_val, y_val_k))
#            x_val2 = np.concatenate((x_val2, x_val_k2))
#            y_val2 = np.concatenate((y_val2, y_val_k2))
            vv, park, veg, wat, soil= None, None,None,None,None 
            num +=1

    return x_train, y_train, x_val, y_val, x_train2, y_train2, x_val2, y_val2#, tren, vald,

def combinations_input(x_trai, y_trai, x_va, y_va,x_trai2, y_trai2, x_va2, y_va2, comb,num):
    N = num
    
    x_train = np.ndarray(shape=(x_trai.shape[0], x_trai.shape[1], x_trai.shape[2], N), dtype='float32')
    y_train = np.ndarray(shape=(y_trai.shape[0], y_trai.shape[1],y_trai.shape[2], y_trai.shape[3]), dtype='float32')
    x_val = np.ndarray(shape=(x_va.shape[0], x_va.shape[1], x_va.shape[2], N), dtype='float32')
    y_val = np.ndarray(shape=(y_va.shape[0], y_va.shape[1],y_va.shape[2], y_va.shape[3]), dtype='float32')

    x_train2 = np.ndarray(shape=(x_trai2.shape[0], x_trai2.shape[1], x_trai2.shape[2], N), dtype='float32')
    y_train2 = np.ndarray(shape=(y_trai2.shape[0], y_trai2.shape[1],y_trai2.shape[2], y_trai.shape[3]), dtype='float32')
    x_val2 = np.ndarray(shape=(x_va2.shape[0], x_va2.shape[1], x_va2.shape[2], N), dtype='float32')
    y_val2 = np.ndarray(shape=(y_va2.shape[0], y_va2.shape[1],y_va2.shape[2], y_va2.shape[3]), dtype='float32')


    if comb == "VV":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,0], y_trai, x_va[:,:,:,0], y_va,x_trai2[:,:,:,0], y_trai2, x_va2[:,:,:,0], y_va2
        x_train, x_val,x_train2, x_val2 = np.expand_dims(x_train, 3),  np.expand_dims(x_val, 3),  np.expand_dims(x_train2, 3), np.expand_dims(x_val2, 3)
    elif comb == "VH":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,1], y_trai, x_va[:,:,:,1], y_va,x_trai2[:,:,:,1], y_trai2, x_va2[:,:,:,1], y_va2
        x_train, x_val,x_train2, x_val2 = np.expand_dims(x_train, 3),  np.expand_dims(x_val, 3),  np.expand_dims(x_train2, 3), np.expand_dims(x_val2, 3)
    elif comb == "VVaVH":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,0:2], y_trai, x_va[:,:,:,0:2], y_va,x_trai2[:,:,:,0:2], y_trai2, x_va2[:,:,:,0:2], y_va2
    elif comb == "VVaVHaRatio":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,0:3], y_trai, x_va[:,:,:,0:3], y_va,x_trai2[:,:,:,0:3], y_trai2, x_va2[:,:,:,0:3], y_va2
    elif comb == "VVaVHaSum":
        x_train[:,:,:,0:2], y_train, x_val[:,:,:,0:2], y_val,x_train2[:,:,:,0:2], y_train2, x_val2[:,:,:,0:2], y_val2 = x_trai[:,:,:,0:2], y_trai, x_va[:,:,:,0:2], y_va,x_trai2[:,:,:,0:2], y_trai2, x_va2[:,:,:,0:2], y_va2
        x_train[:,:,:,2], x_val[:,:,:,2],x_train2[:,:,:,2], x_val2[:,:,:,2]= x_trai[:,:,:,3], x_va[:,:,:,3],x_trai2[:,:,:,3], x_va2[:,:,:,3]
    elif comb == "VVaVHaDiff":
        x_train[:,:,:,0:2], y_train, x_val[:,:,:,0:2], y_val,x_train2[:,:,:,0:2], y_train2, x_val2[:,:,:,0:2], y_val2 = x_trai[:,:,:,0:2], y_trai, x_va[:,:,:,0:2], y_va,x_trai2[:,:,:,0:2], y_trai2, x_va2[:,:,:,0:2], y_va2
        x_train[:,:,:,2], x_val[:,:,:,2], x_train2[:,:,:,2], x_val2[:,:,:,2] = x_trai[:,:,:,4], x_va[:,:,:,4], x_trai2[:,:,:,4], x_va2[:,:,:,4]
    elif comb == "Ratio":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,2], y_trai, x_va[:,:,:,2], y_va,x_trai2[:,:,:,2], y_trai2, x_va2[:,:,:,2], y_va2
        x_train, x_val,x_train2, x_val2 = np.expand_dims(x_train, 3),  np.expand_dims(x_val, 3),  np.expand_dims(x_train2, 3), np.expand_dims(x_val2, 3)
    elif comb == "Total":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai, y_trai, x_va, y_va,x_trai2, y_trai2, x_va2, y_va2
    
    return x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2

#def load_data_super_resolution(dataset_folder, patch_side,patch_side_2,indices):
#    
#    #############
#    # short names
#    path, ps, ps2,v = dataset_folder, patch_side,patch_side_2, indices
#    #############
#    dir_list = os.listdir(path)
#    dir_list.sort()
#    print(dir_list)
#    N = 5
#    Out = 3
#    import random
#    num = 1
#    x_train = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
#    y_train = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
#    x_val = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
#    y_val = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
#    
#    x_train2 = np.ndarray(shape=(0, ps2, ps2, N), dtype='float32')
#    y_train2 = np.ndarray(shape=(0, ps2,ps2, Out), dtype='float32')
#    x_val2 = np.ndarray(shape=(0, ps2, ps2, N), dtype='float32')
#    y_val2 = np.ndarray(shape=(0, ps2,ps2, Out), dtype='float32')
#    
##    tren = np.zeros(shape=(2800,2700),dtype='float32')
##    vald = np.zeros(shape=(2800,2700),dtype='float32')
#    r = 16
#    for file in dir_list:
#        if file.lower().find('albufera_vv.tif') != -1 and file[0]==num: #and file[2]==str(date[num]) and date[num] < 10: #and file[2]<str(7):
#            vv_file = file
#            print(vv_file)
#            park_file ='park' + vv_file[-4:]
#            veg_file =  str(num) + '_vegetation' + vv_file[-4:]
#            wat_file =  str(num) + '_water' + vv_file[-4:]
#            soil_file =  str(num) + '_bare_soil' + vv_file[-4:]
##            soil_file_2 =  str(num) + '_cloud_low_proba' + vv_file[-4:]
#            
#            dataset = gdal.Open(path + veg_file, gdal.GA_ReadOnly)
#            veg = dataset.ReadAsArray()
#            dataset = None
#            dataset = gdal.Open(path + wat_file, gdal.GA_ReadOnly)
#            wat = dataset.ReadAsArray()
#            dataset = None
#            dataset = gdal.Open(path + soil_file, gdal.GA_ReadOnly)
#            soil = dataset.ReadAsArray()
#            dataset = None
##            dataset = gdal.Open(path + soil_file_2, gdal.GA_ReadOnly)
##            soil_2 = dataset.ReadAsArray()
##            dataset = None
##            soil = soil + soil_2
#            
#            vh_file =  str(num) + '_Albufera_VH' + vv_file[-4:]
#            dataset = gdal.Open(path + vh_file, gdal.GA_ReadOnly)
#            vh = dataset.ReadAsArray()
#            dataset = None
#
#            dataset = gdal.Open(path + vv_file, gdal.GA_ReadOnly)
#            vv = dataset.ReadAsArray()
#            dataset = None
#            dataset = gdal.Open(path + park_file, gdal.GA_ReadOnly)
#            park = dataset.ReadAsArray()
#            dataset = None
#            print(max(veg))
#            veg = veg[0,:,:]/255
#            soil = soil[0,:,:]/255
#            wat = wat[0,:,:]/255
#            park = 1 - park/255
#            vv = vv[0,:,:]
#            vh = vh[0,:,:]
##### other additional input    
#            [s1, s2] = vv.shape
#            vv_vh = np.zeros(shape=(s1,s2)) #vv_vh[0,:,:]
#            ave_vv_vh = np.zeros(shape=(s1,s2))#ave_vv_vh[0,:,:]
#            diff_vv_vh = np.zeros(shape=(s1,s2))#diff_vv_vh[0,:,:]
#
#            p2 = []
#            for y in range(1,s1-ps+1,r): 
#                for x in range(1,s2-ps+1,r):
#                    mask_d0 = park[y:y+ps,x:x+ps]
#                    [m1,m2] = mask_d0.shape
#                    s_0 =  mask_d0.sum()
#                    if s_0 == 0:
#                        p2.append([y,x])
#
#            p = p2#[p2[s] for s in v]                    
##            random.shuffle(p)
#            P = int(2000)
#            p_train,p_val= p[:int(0.8*P)],p[int(0.8*P):P]
#            print(len(p_train))
#            print(len(p_val))
#
#            x_train_k = np.ndarray(shape=(len(p_train), ps, ps, N), dtype='float32')
#            y_train_k = np.ndarray(shape=(len(p_train), ps, ps, Out), dtype='float32')
#            x_train_k2 = np.ndarray(shape=(len(p_train), ps2, ps2, N), dtype='float32')
#            y_train_k2 = np.ndarray(shape=(len(p_train), ps2, ps2, Out), dtype='float32')
#            n = 0
#            for patch in p_train:
#                y0, x0 = patch[0], patch[1]
#                x_train_k[n,:,:,0] = vv[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,:,:,1] = vh[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,:,:,2] = vv_vh[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,:,:,3] = ave_vv_vh[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,:,:,4] = diff_vv_vh[y0:y0+ps,x0:x0+ps]
##                tren[y0:y0+ps,x0:x0+ps] += 1
#                
#                y_train_k[n,:,:,0]= soil[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_train_k[n,:,:,1] = veg[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_train_k[n,:,:,2] = wat[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                
#                x_train_k2[n,:,:,0] = vv[y0:y0+ps2,x0:x0+ps2]
#                x_train_k2[n,:,:,1] = vh[y0:y0+ps2,x0:x0+ps2]
#                x_train_k2[n,:,:,2] = vv_vh[y0:y0+ps2,x0:x0+ps2]
#                x_train_k2[n,:,:,3] = ave_vv_vh[y0:y0+ps2,x0:x0+ps2]
#                x_train_k2[n,:,:,4] = diff_vv_vh[y0:y0+ps2,x0:x0+ps2]
##                tren[y0:y0+ps,x0:x0+ps] += 1
#                
#                y_train_k2[n,:,:,0]= soil[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_train_k2[n,:,:,1] = veg[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_train_k2[n,:,:,2] = wat[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                n = n + 1
#            x_train = np.concatenate((x_train, x_train_k))
#            y_train = np.concatenate((y_train, y_train_k))
#            x_train2 = np.concatenate((x_train2, x_train_k2))
#            y_train2 = np.concatenate((y_train2, y_train_k2))
#
#
#            x_val_k = np.ndarray(shape=(len(p_val), ps, ps, N), dtype='float32')
#            y_val_k = np.ndarray(shape=(len(p_val), ps, ps, Out), dtype='float32')
#            x_val_k2 = np.ndarray(shape=(len(p_val), ps2, ps2, N), dtype='float32')
#            y_val_k2 = np.ndarray(shape=(len(p_val), ps2, ps2, Out), dtype='float32')
#            
#            n = 0
#            for patch in p_val:
#                y0, x0 = patch[0], patch[1]
#                x_val_k[n,:,:,0] = vv[y0:y0+ps,x0:x0+ps]
#                x_val_k[n,:,:,1] = vh[y0:y0+ps,x0:x0+ps]
#                x_val_k[n,:,:,2] = vv_vh[y0:y0+ps,x0:x0+ps]
#                x_val_k[n,:,:,3] = ave_vv_vh[y0:y0+ps,x0:x0+ps]
#                x_val_k[n,:,:,4] = diff_vv_vh[y0:y0+ps,x0:x0+ps]
##                vald[y0:y0+ps,x0:x0+ps] += 1
#                y_val_k[n,:,:,0] = soil[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_val_k[n,:,:,1] = veg[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_val_k[n,:,:,2] = wat[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                
#                x_val_k2[n,:,:,0] = vv[y0:y0+ps2,x0:x0+ps2]
#                x_val_k2[n,:,:,1] = vh[y0:y0+ps2,x0:x0+ps2]
#                x_val_k2[n,:,:,2] = vv_vh[y0:y0+ps2,x0:x0+ps2]
#                x_val_k2[n,:,:,3] = ave_vv_vh[y0:y0+ps2,x0:x0+ps2]
#                x_val_k2[n,:,:,4] = diff_vv_vh[y0:y0+ps2,x0:x0+ps2]
##                vald[y0:y0+ps,x0:x0+ps] += 1
#                y_val_k2[n,:,:,0] = soil[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_val_k2[n,:,:,1] = veg[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_val_k2[n,:,:,2] = wat[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                
#                n = n + 1
#            x_val = np.concatenate((x_val, x_val_k))
#            y_val = np.concatenate((y_val, y_val_k))
#            x_val2 = np.concatenate((x_val2, x_val_k2))
#            y_val2 = np.concatenate((y_val2, y_val_k2))
#            vv, park, veg, wat, soil= None, None,None,None,None 
#            num +=1
#
#    return x_train, y_train, x_val, y_val, x_train2, y_train2, x_val2, y_val2#, tren, vald,def load_data(dataset_folder, patch_side,patch_side_2,indices,indices1):
    
    #############
    # short names
def load_data(dataset_folder, patch_side,patch_side_2,indices,indices1, date):
    
    #############
    # short names
    path, ps, ps2,v,v1 = dataset_folder, patch_side,patch_side_2, indices,indices1
    #############
    dir_list = os.listdir(path)
    dir_list.sort()
    print(dir_list)
    N = 5
    Out = 3
    import random
    num = date + 1
    x_train = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
    y_train = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
    x_val = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
    y_val = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
    
    x_train2 = np.ndarray(shape=(0, ps2, ps2, N), dtype='float32')
    y_train2 = np.ndarray(shape=(0, ps2,ps2, Out), dtype='float32')
    x_val2 = np.ndarray(shape=(0, ps2, ps2, N), dtype='float32')
    y_val2 = np.ndarray(shape=(0, ps2,ps2, Out), dtype='float32')
    
#    tren = np.zeros(shape=(2800,2700),dtype='float32')
#    vald = np.zeros(shape=(2800,2700),dtype='float32')
    r = 32
    for file in dir_list:
        print(file)
        if file.lower().find('albufera_vv.tif') != -1 and file[0]==str(num) :# and num < 2:
            vv_file = file
            print(vv_file)
            park_file ='park' + vv_file[-4:] #str(num) + '_
            veg_file =  str(num) + '_vegetation' + vv_file[-4:]
            wat_file =  str(num) + '_water' + vv_file[-4:]
            soil_file =  str(num) + '_bare_soil' + vv_file[-4:]
            patch_file =  'Patches_0' + vv_file[-4:]
            dataset = gdal.Open(path + patch_file, gdal.GA_ReadOnly)
            pat0 = dataset.ReadAsArray()
            dataset = None
            patch_file =  'Patches_1' + vv_file[-4:]
            dataset = gdal.Open(path + patch_file, gdal.GA_ReadOnly)
            pat1 = dataset.ReadAsArray()
            dataset = None
            patch_file =  'Patches_2' + vv_file[-4:]
            dataset = gdal.Open(path + patch_file, gdal.GA_ReadOnly)
            pat2 = dataset.ReadAsArray()
            dataset = None
            patch_file =  'Patches_3' + vv_file[-4:]
            dataset = gdal.Open(path + patch_file, gdal.GA_ReadOnly)
            pat3 = dataset.ReadAsArray()
            dataset = None
            patch_file =  'Patches_4' + vv_file[-4:]
            dataset = gdal.Open(path + patch_file, gdal.GA_ReadOnly)
            pat4 = dataset.ReadAsArray()
            dataset = None
#            soil_file_2 =  str(num) + '_cloud_low_proba' + vv_file[-4:]
            
            dataset = gdal.Open(path + veg_file, gdal.GA_ReadOnly)
            veg = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path + wat_file, gdal.GA_ReadOnly)
            wat = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path + soil_file, gdal.GA_ReadOnly)
            soil = dataset.ReadAsArray()
            dataset = None
#            dataset = gdal.Open(path + soil_file_2, gdal.GA_ReadOnly)
#            soil_2 = dataset.ReadAsArray()
#            dataset = None
#            soil = soil + soil_2
            
            vh_file =  str(num) + '_Albufera_VH' + vv_file[-4:]
            dataset = gdal.Open(path + vh_file, gdal.GA_ReadOnly)
            vh = dataset.ReadAsArray()
            dataset = None
            vv_file =  str(num) + '_Albufera_VV' + vv_file[-4:]
            dataset = gdal.Open(path + vv_file, gdal.GA_ReadOnly)
            vv = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path + park_file, gdal.GA_ReadOnly)
            park = dataset.ReadAsArray()
            dataset = None
            veg = veg[0,:,:]/255
            soil = soil[0,:,:]/255
            wat = wat[0,:,:]/255
            park = 1 - park/255
            pato = pat0*pat1*pat2*pat3*pat4
#            pat0 = 1-pat0
#            pat1 = 1-pat1
#            pat2 = 1-pat2
#            pat3 = 1-pat3
#            pat4 = 1-pat4
            park = park + (1 - pato)
            imsave(path + park_file[:-4] + "_quellousato.tif",park)
            vv = vv[0,:,:]
            vh = vh[0,:,:]
            back = 1 - veg*wat*soil
#### other additional input    
            [s1, s2] = vv.shape
            vv_vh = (vv + 10**(-10))/(vh + 10**(-10))  #vv_vh[0,:,:]
            ave_vv_vh =  (vv + vh)/2 #np.zeros(shape=(s1,s2))#ave_vv_vh[0,:,:]
            diff_vv_vh = vv - vh #diff_vv_vh[0,:,:]
            p2 = []
            print(len(p2))
            for y in range(1,s1-ps+1,r): 
                for x in range(1,s2-ps+1,r):
                    mask_d0 = park[y:y+ps,x:x+ps]
                    mask_veg = veg[y:y+ps,x:x+ps]
                    mask_wat = wat[y:y+ps,x:x+ps]
                    mask_soil = soil[y:y+ps,x:x+ps]
                    [m1,m2] = mask_d0.shape
                    s_0 =  mask_d0.sum()
                    mask_T= mask_veg + mask_wat + mask_soil
                    s_T =  mask_T.sum()
                    if s_0 == 0 and s_T > (0.75)*(m1*m2):
                        p2.append([y,x])
            p_train = []
            p_val = []
            print(len(p2))
            ### TEST
            adic = {}
            adic = v['dic']
            indi = adic['p']

            adic1 = {}
            adic1 = v1['dic']
            indi1 = adic1['p']
###############
#            if len(p2) > 800: 
#    
#                p = [p2[s] for s in range(0,len(p2),50)]#,int((len(p2) - 800)))]
#            else: 
#                p = [p2[s] for s in range(0,700,50)]
#            print(p)
#            p_val = p
            ######
#            p = [p2[s] for s in indi]
            ########
#            if num == 1:
#                p_val = p[:int(0.2*P)]
#                
#            else: 
#                p_train = p[:int(0.8*P)]
            
#############            
#            p =[p2[v[str(s)]] for s in range(700)]  # p2#[v[s] for s in range(len(p2))]                    
    ########              
            P = len(p2)#3000#len(p2)#int(3000)
            p = p2#[p2[s] for s in v]  
            random.shuffle(p)
            p_train,p_val= p[:int(0.9*P)],p[int(0.9*P):P]
            # print(len(p_train))
            # print(len(p_val))

            x_train_k = np.ndarray(shape=(len(p_train), ps, ps, N), dtype='float32')
            y_train_k = np.ndarray(shape=(len(p_train), ps, ps, Out), dtype='float32')
#            x_train_k2 = np.ndarray(shape=(len(p_train), ps2, ps2, N), dtype='float32')
#            y_train_k2 = np.ndarray(shape=(len(p_train), ps2, ps2, Out), dtype='float32')
            n = 0
            for patch in p_train:
                y0, x0 = patch[0], patch[1]
                x_train_k[n,:,:,0] = vv[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,1] = vh[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,2] = vv_vh[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,3] = ave_vv_vh[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,4] = diff_vv_vh[y0:y0+ps,x0:x0+ps]
#                tren[y0:y0+ps,x0:x0+ps] += 1
                
                y_train_k[n,:,:,0]= soil[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_train_k[n,:,:,1] = veg[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_train_k[n,:,:,2] = wat[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_train_k[n,:,:,3] = back[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                x_train_k2[n,:,:,0] = vv[y0:y0+ps2,x0:x0+ps2]
#                x_train_k2[n,:,:,1] = vh[y0:y0+ps2,x0:x0+ps2]
#                x_train_k2[n,:,:,2] = vv_vh[y0:y0+ps2,x0:x0+ps2]
#                x_train_k2[n,:,:,3] = ave_vv_vh[y0:y0+ps2,x0:x0+ps2]
#                x_train_k2[n,:,:,4] = diff_vv_vh[y0:y0+ps2,x0:x0+ps2]
##                tren[y0:y0+ps,x0:x0+ps] += 1
#                
#                y_train_k2[n,:,:,0]= soil[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_train_k2[n,:,:,1] = veg[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_train_k2[n,:,:,2] = wat[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
##                y_train_k2[n,:,:,3] = back[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                n = n + 1
            x_train = np.concatenate((x_train, x_train_k))
            y_train = np.concatenate((y_train, y_train_k))
#            x_train2 = np.concatenate((x_train2, x_train_k2))
#            y_train2 = np.concatenate((y_train2, y_train_k2))


            x_val_k = np.ndarray(shape=(len(p_val), ps, ps, N), dtype='float32')
            y_val_k = np.ndarray(shape=(len(p_val), ps, ps, Out), dtype='float32')
#            x_val_k2 = np.ndarray(shape=(len(p_val), ps2, ps2, N), dtype='float32')
#            y_val_k2 = np.ndarray(shape=(len(p_val), ps2, ps2, Out), dtype='float32')
            
            n = 0
            for patch in p_val:
                y0, x0 = patch[0], patch[1]
                x_val_k[n,:,:,0] = vv[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,1] = vh[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,2] = vv_vh[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,3] = ave_vv_vh[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,4] = diff_vv_vh[y0:y0+ps,x0:x0+ps]
#                vald[y0:y0+ps,x0:x0+ps] += 1
                y_val_k[n,:,:,0] = soil[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_val_k[n,:,:,1] = veg[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_val_k[n,:,:,2] = wat[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_val_k[n,:,:,3] = back[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                x_val_k2[n,:,:,0] = vv[y0:y0+ps2,x0:x0+ps2]
#                x_val_k2[n,:,:,1] = vh[y0:y0+ps2,x0:x0+ps2]
#                x_val_k2[n,:,:,2] = vv_vh[y0:y0+ps2,x0:x0+ps2]
#                x_val_k2[n,:,:,3] = ave_vv_vh[y0:y0+ps2,x0:x0+ps2]
#                x_val_k2[n,:,:,4] = diff_vv_vh[y0:y0+ps2,x0:x0+ps2]
##                vald[y0:y0+ps,x0:x0+ps] += 1
#                y_val_k2[n,:,:,0] = soil[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_val_k2[n,:,:,1] = veg[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                y_val_k2[n,:,:,2] = wat[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
##                y_val_k2[n,:,:,3] = back[y0:y0+ps2, x0:x0+ps2]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                n = n + 1
            x_val = np.concatenate((x_val, x_val_k))
            y_val = np.concatenate((y_val, y_val_k))
#            x_val2 = np.concatenate((x_val2, x_val_k2))
#            y_val2 = np.concatenate((y_val2, y_val_k2))
            vv, park, veg, wat, soil= None, None,None,None,None 
            # num +=1
            # else: 

    return x_train, y_train, x_val, y_val, x_train2, y_train2, x_val2, y_val2 #, x_test, y_test #, tren, vald,

def combinations_input2(x_trai, y_trai, x_va, y_va,x_trai2, y_trai2, x_va2, y_va2, comb,num):
    N = num
    
    x_train = np.ndarray(shape=(x_trai.shape[0], x_trai.shape[1], x_trai.shape[2], N), dtype='float32')
    y_train = np.ndarray(shape=(y_trai.shape[0], y_trai.shape[1],y_trai.shape[2], y_trai.shape[3]), dtype='float32')
    x_val = np.ndarray(shape=(x_va.shape[0], x_va.shape[1], x_va.shape[2], N), dtype='float32')
    y_val = np.ndarray(shape=(y_va.shape[0], y_va.shape[1],y_va.shape[2], y_va.shape[3]), dtype='float32')

    x_train2 = np.ndarray(shape=(x_trai2.shape[0], x_trai2.shape[1], x_trai2.shape[2], N), dtype='float32')
    y_train2 = np.ndarray(shape=(y_trai2.shape[0], y_trai2.shape[1],y_trai2.shape[2], y_trai.shape[3]), dtype='float32')
    x_val2 = np.ndarray(shape=(x_va2.shape[0], x_va2.shape[1], x_va2.shape[2], N), dtype='float32')
    y_val2 = np.ndarray(shape=(y_va2.shape[0], y_va2.shape[1],y_va2.shape[2], y_va2.shape[3]), dtype='float32')


    if comb == "VV":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,0], y_trai, x_va[:,:,:,0], y_va,x_trai2[:,:,:,0], y_trai2, x_va2[:,:,:,0], y_va2
        x_train, x_val,x_train2, x_val2 = np.expand_dims(x_train, 3),  np.expand_dims(x_val, 3),  np.expand_dims(x_train2, 3), np.expand_dims(x_val2, 3)
    elif comb == "VH":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,1], y_trai, x_va[:,:,:,1], y_va,x_trai2[:,:,:,1], y_trai2, x_va2[:,:,:,1], y_va2
        x_train, x_val,x_train2, x_val2 = np.expand_dims(x_train, 3),  np.expand_dims(x_val, 3),  np.expand_dims(x_train2, 3), np.expand_dims(x_val2, 3)
    elif comb == "VVaVH":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,0:2], y_trai, x_va[:,:,:,0:2], y_va,x_trai2[:,:,:,0:2], y_trai2, x_va2[:,:,:,0:2], y_va2
    elif comb == "VVaVHaRatio":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,0:3], y_trai, x_va[:,:,:,0:3], y_va,x_trai2[:,:,:,0:3], y_trai2, x_va2[:,:,:,0:3], y_va2
    elif comb == "VVaVHaSum":
        x_train[:,:,:,0:2], y_train, x_val[:,:,:,0:2], y_val,x_train2[:,:,:,0:2], y_train2, x_val2[:,:,:,0:2], y_val2 = x_trai[:,:,:,0:2], y_trai, x_va[:,:,:,0:2], y_va,x_trai2[:,:,:,0:2], y_trai2, x_va2[:,:,:,0:2], y_va2
        x_train[:,:,:,2], x_val[:,:,:,2],x_train2[:,:,:,2], x_val2[:,:,:,2]= x_trai[:,:,:,3], x_va[:,:,:,3],x_trai2[:,:,:,3], x_va2[:,:,:,3]
    elif comb == "VVaVHaDiff":
        x_train[:,:,:,0:2], y_train, x_val[:,:,:,0:2], y_val,x_train2[:,:,:,0:2], y_train2, x_val2[:,:,:,0:2], y_val2 = x_trai[:,:,:,0:2], y_trai, x_va[:,:,:,0:2], y_va,x_trai2[:,:,:,0:2], y_trai2, x_va2[:,:,:,0:2], y_va2
        x_train[:,:,:,2], x_val[:,:,:,2], x_train2[:,:,:,2], x_val2[:,:,:,2] = x_trai[:,:,:,4], x_va[:,:,:,4], x_trai2[:,:,:,4], x_va2[:,:,:,4]
    elif comb == "Ratio":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,2], y_trai, x_va[:,:,:,2], y_va,x_trai2[:,:,:,2], y_trai2, x_va2[:,:,:,2], y_va2
        x_train, x_val,x_train2, x_val2 = np.expand_dims(x_train, 3),  np.expand_dims(x_val, 3),  np.expand_dims(x_train2, 3), np.expand_dims(x_val2, 3)
    elif comb == "Total":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai, y_trai, x_va, y_va,x_trai2, y_trai2, x_va2, y_va2
    
    return x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2

def combinations_input_super_resolution(x_trai, y_trai, x_va, y_va,x_trai2, y_trai2, x_va2, y_va2, comb,num):
    N = num
    
    x_train = np.ndarray(shape=(x_trai.shape[0], x_trai.shape[1], x_trai.shape[2], N), dtype='float32')
    y_train = np.ndarray(shape=(y_trai.shape[0], y_trai.shape[1],y_trai.shape[2], y_trai.shape[3]), dtype='float32')
    x_val = np.ndarray(shape=(x_va.shape[0], x_va.shape[1], x_va.shape[2], N), dtype='float32')
    y_val = np.ndarray(shape=(y_va.shape[0], y_va.shape[1],y_va.shape[2], y_va.shape[3]), dtype='float32')

    x_train2 = np.ndarray(shape=(x_trai2.shape[0], x_trai2.shape[1], x_trai2.shape[2], N), dtype='float32')
    y_train2 = np.ndarray(shape=(y_trai2.shape[0], y_trai2.shape[1],y_trai2.shape[2], y_trai.shape[3]), dtype='float32')
    x_val2 = np.ndarray(shape=(x_va2.shape[0], x_va2.shape[1], x_va2.shape[2], N), dtype='float32')
    y_val2 = np.ndarray(shape=(y_va2.shape[0], y_va2.shape[1],y_va2.shape[2], y_va2.shape[3]), dtype='float32')


    if comb == "VV":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,0], y_trai, x_va[:,:,:,0], y_va,x_trai2[:,:,:,0], y_trai2, x_va2[:,:,:,0], y_va2
        x_train, x_val,x_train2, x_val2 = np.expand_dims(x_train, 3),  np.expand_dims(x_val, 3),  np.expand_dims(x_train2, 3), np.expand_dims(x_val2, 3)
    elif comb == "VH":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,1], y_trai, x_va[:,:,:,1], y_va,x_trai2[:,:,:,1], y_trai2, x_va2[:,:,:,1], y_va2
        x_train, x_val,x_train2, x_val2 = np.expand_dims(x_train, 3),  np.expand_dims(x_val, 3),  np.expand_dims(x_train2, 3), np.expand_dims(x_val2, 3)
    elif comb == "VVaVH":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,0:2], y_trai, x_va[:,:,:,0:2], y_va,x_trai2[:,:,:,0:2], y_trai2, x_va2[:,:,:,0:2], y_va2
    elif comb == "VVaVHaRatio":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,0:3], y_trai, x_va[:,:,:,0:3], y_va,x_trai2[:,:,:,0:3], y_trai2, x_va2[:,:,:,0:3], y_va2
    elif comb == "VVaVHaSum":
        x_train[:,:,:,0:2], y_train, x_val[:,:,:,0:2], y_val,x_train2[:,:,:,0:2], y_train2, x_val2[:,:,:,0:2], y_val2 = x_trai[:,:,:,0:2], y_trai, x_va[:,:,:,0:2], y_va,x_trai2[:,:,:,0:2], y_trai2, x_va2[:,:,:,0:2], y_va2
        x_train[:,:,:,2], x_val[:,:,:,2],x_train2[:,:,:,2], x_val2[:,:,:,2]= x_trai[:,:,:,3], x_va[:,:,:,3],x_trai2[:,:,:,3], x_va2[:,:,:,3]
    elif comb == "VVaVHaDiff":
        x_train[:,:,:,0:2], y_train, x_val[:,:,:,0:2], y_val,x_train2[:,:,:,0:2], y_train2, x_val2[:,:,:,0:2], y_val2 = x_trai[:,:,:,0:2], y_trai, x_va[:,:,:,0:2], y_va,x_trai2[:,:,:,0:2], y_trai2, x_va2[:,:,:,0:2], y_va2
        x_train[:,:,:,2], x_val[:,:,:,2], x_train2[:,:,:,2], x_val2[:,:,:,2] = x_trai[:,:,:,4], x_va[:,:,:,4], x_trai2[:,:,:,4], x_va2[:,:,:,4]
    elif comb == "Ratio":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,2], y_trai, x_va[:,:,:,2], y_va,x_trai2[:,:,:,2], y_trai2, x_va2[:,:,:,2], y_va2
        x_train, x_val,x_train2, x_val2 = np.expand_dims(x_train, 3),  np.expand_dims(x_val, 3),  np.expand_dims(x_train2, 3), np.expand_dims(x_val2, 3)
    elif comb == "Total":
        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai, y_trai, x_va, y_va,x_trai2, y_trai2, x_va2, y_va2
    
    return x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2

#
#
#
#
#def load_dataset(dataset_folder, patch_side, border_width,identity,num_bands, city, patch):#,ff): #, output_folder):
#    
#    #############
#    # short names
#    path, ps, r,city_name,patch0 = dataset_folder, patch_side, border_width,city, patch
#    #output_path = output_folder
#    print(r)
#    #############
#    k = identity
#    dir_list = os.listdir(path)
#    dir_list.sort()
##    dir_list = dir_list[:-2]
##    print(dir_list)
#    N = num_bands#-4
#    Out = 6+4 
#    import random
#    Ts = ps #2*ps when I want to consider full resolution (10-m)
#    Ts_2 = ps//2
##    k_L = num
#    x_train = np.ndarray(shape=(0, Ts_2, Ts_2, N), dtype='float32')
#    y_train = np.ndarray(shape=(0, Ts_2, Ts_2, Out), dtype='float32')
#    x_val = np.ndarray(shape=(0, Ts_2, Ts_2, N), dtype='float32')
#    y_val = np.ndarray(shape=(0, Ts_2, Ts_2, Out), dtype='float32')
##    K = 33 # 17# #100 #300
#    K1 = Ts
#    K2 = Ts
#    K3 = Ts_2
#    K4 = Ts_2
#    x_test2 = np.ndarray(shape=(0, Ts , Ts,N), dtype='float32')
#    y_test2 = np.ndarray(shape=(0, Ts , Ts,Out-5), dtype='float32')
#    x_test = np.ndarray(shape=(0, Ts_2 , Ts_2,N), dtype='float32')
#    y_test = np.ndarray(shape=(0, Ts_2 , Ts_2,Out-5), dtype='float32')
#    num = 0 
##    for num1 in range(1):#dir_list:
##        file1 = dir_list[num1]
##        if file1.find(city_name) != -1 and file1.find("large") != -1: # and num == k_L and file[2]<str(7):
#    vh_file = dir_list[0]
##    print(vh_file)
#    vh1_file =city_name + '_large_' + str(patch0) + '_B08D'+ vh_file[-4:]
#    vh2_file =city_name + '_large_' + str(patch0) + '_B04D'+ vh_file[-4:]
#    vh4_file =city_name + '_large_' + str(patch0) + '_B03D'+ vh_file[-4:]
#    vh5_file =city_name + '_large_' + str(patch0) + '_B02D'+ vh_file[-4:]
##            b5 b6 b7 b8a b11 b12  
#    
#    vh6_file =city_name + '_large_' + str(patch0) + '_B'+k+'DR'+ vh_file[-4:]
#    vh6f_file =city_name + '_large_' + str(patch0) + '_B'+k+'DH'+ vh_file[-4:]
#    vh6r1_file =city_name + '_large_' + str(patch0) + '_B'+k+'R1'+ vh_file[-4:]
#    vh7_file =city_name + '_large_' + str(patch0) + '_B'+k+ vh_file[-4:]
#    dataset = gdal.Open(path + vh6r1_file, gdal.GA_ReadOnly)
#    b6_r1= dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None
#
#    
#    
#    dataset = gdal.Open(path + vh1_file, gdal.GA_ReadOnly)
#    b8_d= dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None
#    
#    dataset = gdal.Open(path + vh2_file, gdal.GA_ReadOnly)
#    b4_d = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None           
#    
#    
#    dataset = gdal.Open(path + vh4_file, gdal.GA_ReadOnly)      
#    b3_d = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None
##            
#    dataset = gdal.Open(path + vh5_file, gdal.GA_ReadOnly)      
#    b2_d = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None
#    
#    
#    dataset = gdal.Open(path + vh7_file, gdal.GA_ReadOnly)
#    b6 = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None           
#    
#    dataset = gdal.Open(path + vh6_file, gdal.GA_ReadOnly)
#    b6_r= dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None
#    
#    dataset = gdal.Open(path + vh6f_file, gdal.GA_ReadOnly)
#    b6_hh= dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None
#    vh1_file =city_name + '_large_' + str(patch0) + '_B02H1'+ vh_file[-4:] # senza H1 per immagine normale
#    vh2_file = city_name + '_large_' + str(patch0) + '_B03H1'+ vh_file[-4:]
#    vh3_file =city_name + '_large_' + str(patch0) + '_B04H1'+ vh_file[-4:]
#    vh4_file =city_name + '_large_' + str(patch0) + '_B08H1'+ vh_file[-4:]
#    
#    dataset = gdal.Open(path + vh1_file, gdal.GA_ReadOnly)
#    b2_h1 = dataset.ReadAsArray()
##            b2_h1 = b2_h1/(2**16)
#    dataset = None
#    
#    dataset = gdal.Open(path + vh2_file, gdal.GA_ReadOnly)
#    b3_h1 = dataset.ReadAsArray()
##            b3_h1 = b3_h1/(2**16)
#    dataset = None           
#    
#    dataset = gdal.Open(path+vh3_file, gdal.GA_ReadOnly)
#    b4_h1 = dataset.ReadAsArray()
##            b4_h1 = b4_h1/(2**16)
#    dataset = None         
#    
#    dataset = gdal.Open(path + vh4_file, gdal.GA_ReadOnly)      
#    b8_h1 = dataset.ReadAsArray()
##            b8_h1 = b8_h1/(2**16)
#    dataset = None
#    vh1_file =city_name + '_large_' + str(patch0) + '_B05H1'+ vh_file[-4:] # senza H1 per immagine normale
#    vh2_file = city_name + '_large_' + str(patch0) + '_B06H1'+ vh_file[-4:]
#    vh3_file =city_name + '_large_' + str(patch0) + '_B07H1'+ vh_file[-4:]
#    vh4_file =city_name + '_large_' + str(patch0) + '_B8AH1'+ vh_file[-4:]
#    vh5_file = city_name + '_large_' + str(patch0) + '_B11H1'+ vh_file[-4:]
#    vh6_file =city_name + '_large_' + str(patch0) + '_B12H1'+ vh_file[-4:]
#
#    dataset = gdal.Open(path + vh1_file, gdal.GA_ReadOnly)
#    b5_h1 = dataset.ReadAsArray()
##            b9 = b8/(2**16)
#    dataset = None
#    
#    dataset = gdal.Open(path + vh2_file, gdal.GA_ReadOnly)
#    b6_h1 = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None           
#    
#    dataset = gdal.Open(path+vh3_file, gdal.GA_ReadOnly)
#    b7_h1 = dataset.ReadAsArray()
##            b11 = b11/(2**16)
#    dataset = None         
#    
#    dataset = gdal.Open(path + vh4_file, gdal.GA_ReadOnly)      
#    b8_a_h1 = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None
##            
#    dataset = gdal.Open(path+vh5_file, gdal.GA_ReadOnly)
#    b11_h1 = dataset.ReadAsArray()
##            b11 = b11/(2**16)
#    dataset = None
#    
#    dataset = gdal.Open(path + vh6_file, gdal.GA_ReadOnly)
#    b12_h1 = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None
#    
#    vh1_file =city_name + '_large_' + str(patch0) + '_B05DR'+ vh_file[-4:] # senza H1 per immagine normale
#    vh2_file = city_name + '_large_' + str(patch0) + '_B06DR'+ vh_file[-4:]
#    vh3_file =city_name + '_large_' + str(patch0) + '_B07DR'+ vh_file[-4:]
#    vh4_file =city_name + '_large_' + str(patch0) + '_B8ADR'+ vh_file[-4:]
#    vh5_file = city_name + '_large_' + str(patch0) + '_B11DR'+ vh_file[-4:]
#    vh6_file =city_name + '_large_' + str(patch0) + '_B12DR'+ vh_file[-4:]
#
#    dataset = gdal.Open(path + vh1_file, gdal.GA_ReadOnly)
#    b5_r = dataset.ReadAsArray()
##            b9 = b8/(2**16)
#    dataset = None
#    
#    dataset = gdal.Open(path + vh2_file, gdal.GA_ReadOnly)
#    b6_rr = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None           
#    
#    dataset = gdal.Open(path+vh3_file, gdal.GA_ReadOnly)
#    b7_r = dataset.ReadAsArray()
##            b11 = b11/(2**16)
#    dataset = None         
#    
#    dataset = gdal.Open(path + vh4_file, gdal.GA_ReadOnly)      
#    b8_ar = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None
##            
#    dataset = gdal.Open(path+vh5_file, gdal.GA_ReadOnly)
#    b11_r = dataset.ReadAsArray()
##            b11 = b11/(2**16)
#    dataset = None
#    
#    dataset = gdal.Open(path + vh6_file, gdal.GA_ReadOnly)
#    b12_r = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None
#
#    vh1_file =city_name + '_large_' + str(patch0) + '_B05DH'+ vh_file[-4:] # senza H1 per immagine normale
#    vh2_file = city_name + '_large_' + str(patch0) + '_B06DH'+ vh_file[-4:]
#    vh3_file =city_name + '_large_' + str(patch0) + '_B07DH'+ vh_file[-4:]
#    vh4_file =city_name + '_large_' + str(patch0) + '_B8ADH'+ vh_file[-4:]
#    vh5_file = city_name + '_large_' + str(patch0) + '_B11DH'+ vh_file[-4:]
#    vh6_file =city_name + '_large_' + str(patch0) + '_B12DH'+ vh_file[-4:]
#
#    dataset = gdal.Open(path + vh1_file, gdal.GA_ReadOnly)
#    b5_h = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None
#    
#    dataset = gdal.Open(path + vh2_file, gdal.GA_ReadOnly)
#    b6_h = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None           
#    
#    dataset = gdal.Open(path+vh3_file, gdal.GA_ReadOnly)
#    b7_h = dataset.ReadAsArray()
##            b11 = b11/(2**16)
#    dataset = None         
#    
#    dataset = gdal.Open(path + vh4_file, gdal.GA_ReadOnly)      
#    b8_a_h = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None
##            
#    dataset = gdal.Open(path+vh5_file, gdal.GA_ReadOnly)
#    b11_h = dataset.ReadAsArray()
##            b11 = b11/(2**16)
#    dataset = None
#    
#    dataset = gdal.Open(path + vh6_file, gdal.GA_ReadOnly)
#    b12_h = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None
#
#
#    
#    vh1_file =city_name + '_large_' + str(patch0) + '_B02DH'+ vh_file[-4:] # senza H1 per immagine normale
#    vh2_file = city_name + '_large_' + str(patch0) + '_B03DH'+ vh_file[-4:]
#    vh3_file =city_name + '_large_' + str(patch0) + '_B04DH'+ vh_file[-4:]
#    vh4_file =city_name + '_large_' + str(patch0) + '_B08DH'+ vh_file[-4:]
#    
#    dataset = gdal.Open(path + vh1_file, gdal.GA_ReadOnly)
#    b2_h = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None
#    
#    dataset = gdal.Open(path + vh2_file, gdal.GA_ReadOnly)
#    b3_h = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None           
#    
#    dataset = gdal.Open(path+vh3_file, gdal.GA_ReadOnly)
#    b4_h = dataset.ReadAsArray()
##            b11 = b11/(2**16)
#    dataset = None         
#    
#    dataset = gdal.Open(path + vh4_file, gdal.GA_ReadOnly)      
#    b8_h = dataset.ReadAsArray()
##            b8 = b8/(2**16)
#    dataset = None
#
#    p_test = [] 
#    mask = np.zeros(b6_r.shape,dtype='float32')
##            train_patches = np.zeros(b6_r.shape,dtype='float32')
##            val_patches = np.zeros(b6_r.shape,dtype='float32')
#
##            mask[int(530/2):int(1000/2),int(4300/2):int(4750/2)]=1 
##            for y in range(int(530/2),int(1000/2),2*ps):
##                for x in range(int(4300/2),int(4750/2),2*ps):
##                    p_test.append([y,x])
#    print(r)
##    b5_h1 = np.pad(b5_h1,((r,r),(r,r)),'reflect') 
##    b6_h1 = np.pad(b6_h1,((r,r),(r,r)),'reflect') 
##    b7_h1 = np.pad(b7_h1,((r,r),(r,r)),'reflect') 
##    b8_a_h1 = np.pad(b8_a_h1,((r,r),(r,r)),'reflect') 
##    b11_h1 = np.pad(b11_h1,((r,r),(r,r)),'reflect') 
##    b12_h1 = np.pad(b12_h1,((r,r),(r,r)),'reflect') 
##    b2_h1 = np.pad(b2_h1,((r,r),(r,r)),'reflect') 
##    b3_h1 = np.pad(b3_h1,((r,r),(r,r)),'reflect') 
##    b4_h1 = np.pad(b4_h1,((r,r),(r,r)),'reflect') 
##    b8_h1 = np.pad(b8_h1,((r,r),(r,r)),'reflect') 
##    b6_r1 = np.pad(b6_r1,((r,r),(r,r)),'reflect') 
##
##
##    b5_h2 = np.pad(b5_h,((r,r),(r,r)),'reflect') 
##    b6_h2 = np.pad(b6_h,((r,r),(r,r)),'reflect') 
##    b7_h2 = np.pad(b7_h,((r,r),(r,r)),'reflect') 
##    b8_a_h2 = np.pad(b8_a_h,((r,r),(r,r)),'reflect') 
##    b11_h2 = np.pad(b11_h,((r,r),(r,r)),'reflect') 
##    b12_h2 = np.pad(b12_h,((r,r),(r,r)),'reflect') 
##    b2_h2 = np.pad(b2_h,((r,r),(r,r)),'reflect') 
##    b3_h2 = np.pad(b3_h,((r,r),(r,r)),'reflect') 
##    b4_h2 = np.pad(b4_h,((r,r),(r,r)),'reflect') 
##    b8_h2 = np.pad(b8_h,((r,r),(r,r)),'reflect') 
##    b6_r2 = np.pad(b6_r,((r,r),(r,r)),'reflect') 
#
#
#    p_test = [0,0]
#    
#    y0, x0 = p_test
#    print((y0,x0))
##            y0 += r
##            x0 += r
#    x_test_k = np.ndarray(shape=(len(p_test)-1, Ts_2 ,Ts_2, N ), dtype='float32')
#    y_test_k = np.ndarray(shape=(len(p_test)-1, Ts_2 ,Ts_2 , Out-5), dtype='float32')
#    x_test_k2 = np.ndarray(shape=(len(p_test)-1, Ts ,Ts , N), dtype='float32')
#    y_test_k2 = np.ndarray(shape=(len(p_test)-1, Ts ,Ts , Out-5), dtype='float32')
#    n = 0
##            for patch in p_test:
##            y0, x0 = patch[0], patch[1]     
#    print(x_test_k2[n,0,:,:].shape)
#    print(b5_h1[y0:y0+K1 ,x0:x0+K2 ].shape)
#    x_test_k2[n,:,:,0] = b5_h1[y0:y0+K1 ,x0:x0+K2 ]
#    x_test_k2[n,:,:,1] = b6_h1[y0:y0+K1 ,x0:x0+K2 ]
#    x_test_k2[n,:,:,2] = b7_h1[y0:y0+K1 ,x0:x0+K2 ]
#    x_test_k2[n,:,:,3] = b8_a_h1[y0:y0+K1 ,x0:x0+K2 ]
#    x_test_k2[n,:,:,4] = b11_h1[y0:y0+K1 ,x0:x0+K2 ]
#    x_test_k2[n,:,:,5] = b12_h1[y0:y0+K1 ,x0:x0+K2 ]
#    x_test_k2[n,:,:,6] = b8_h1[y0:y0+K1 ,x0:x0+K2 ]
#    x_test_k2[n,:,:,7] = b4_h1[y0:y0+K1 ,x0:x0+K2 ]
#    x_test_k2[n,:,:,8] = b3_h1[y0:y0+K1 ,x0:x0+K2 ]           
#    x_test_k2[n,:,:,9] = b2_h1[y0:y0+K1 ,x0:x0+K2 ]
#
#    x_test_k[n,:,:,0] = b5_h[y0:y0+K3 ,x0:x0+K4 ]
#    x_test_k[n,:,:,1] = b6_h[y0:y0+K3 ,x0:x0+K4 ]
#    x_test_k[n,:,:,2] = b7_h[y0:y0+K3 ,x0:x0+K4 ]
#    x_test_k[n,:,:,3] = b8_a_h[y0:y0+K3 ,x0:x0+K4 ]
#    x_test_k[n,:,:,4] = b11_h[y0:y0+K3 ,x0:x0+K4 ]
#    x_test_k[n,:,:,5] = b12_h[y0:y0+K3 ,x0:x0+K4 ]
#    x_test_k[n,:,:,6] = b8_h[y0:y0+K3 ,x0:x0+K4 ]
#    x_test_k[n,:,:,7] = b4_h[y0:y0+K3 ,x0:x0+K4 ]
#    x_test_k[n,:,:,8] = b3_h[y0:y0+K3 ,x0:x0+K4 ]           
#    x_test_k[n,:,:,9] = b2_h[y0:y0+K3 ,x0:x0+K4 ]
#
#
#    y_test_k[n,:,:,0] = b6_r[y0:y0+K3 ,x0:x0+K4 ]
#    y_test_k2[n,:,:,0] = b6_r1[y0:y0+K1 ,x0:x0+K2 ]
#
#    x_test = np.concatenate((x_test, x_test_k))
#    y_test = np.concatenate((y_test, y_test_k))
#    x_test2 = np.concatenate((x_test2, x_test_k2))
#    y_test2 = np.concatenate((y_test2, y_test_k2))
#    [s1, s2] = b6_r.shape
#    p = []
##            for y in range(1,s1-ps+1,r):
##                for x in range(1,s2-ps+1,r):
##                    mask_d0 = mask[y:y+ps,x:x+ps]
##                    s_0 =  mask_d0.sum()
##                    if s_0 == 0:
##                        p.append([y,x])
#    
#    p = [0,0]
#    y0, x0 = p 
##            y0 += r
##            x0 += r
#
#    p_train = p 
#    p_val = p 
#    
#    x_train_k = np.ndarray(shape=(len(p_train)-1, Ts_2, Ts_2, N), dtype='float32')
#    y_train_k = np.ndarray(shape=(len(p_train)-1, Ts_2, Ts_2, Out), dtype='float32')
#    n = 0
##            for patch in p_train:
##                y0, x0 = patch[0], patch[1]
#    x_train_k[n,:,:,0] = b5_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_train_k[n,:,:,1] = b6_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_train_k[n,:,:,2] = b7_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_train_k[n,:,:,3] = b8_a_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_train_k[n,:,:,4] = b11_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_train_k[n,:,:,5] = b12_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_train_k[n,:,:,6] = b8_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_train_k[n,:,:,7] = b4_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_train_k[n,:,:,8] = b3_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_train_k[n,:,:,9] = b2_h[y0:y0+Ts_2,x0:x0+Ts_2]
#
##                x_train_k[n,0,:,:] = b5_r[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,1,:,:] = b6_rr[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,2,:,:] = b7_r[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,3,:,:] = b8_ar[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,4,:,:] = b11_r[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,5,:,:] = b12_r[y0:y0+ps,x0:x0+ps]
##
##                x_train_k[n,0,:,:] = b6_r[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,1,:,:] = b8_d[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,2,:,:] = b4_d[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,3,:,:] = b3_d[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,4,:,:] = b2_d[y0:y0+ps,x0:x0+ps]
##
##                x_train_k[n,0,:,:] = b6_hh[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,1,:,:] = b8_h[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,2,:,:] = b4_h[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,3,:,:] = b3_h[y0:y0+ps,x0:x0+ps]
##                x_train_k[n,4,:,:] = b2_h[y0:y0+ps,x0:x0+ps]
#
#        
#    y_train_k[n, :, :, 0] = b6[y0:y0+Ts_2,x0:x0+Ts_2]#-b6_r[y0:y0+Ts_2,x0:x0+Ts_2]
#    y_train_k[n, :, :, 1] = b6_r[y0:y0+Ts_2,x0:x0+Ts_2]
#    y_train_k[n,:,:,2] = b8_d[y0:y0+Ts_2,x0:x0+Ts_2]
#    y_train_k[n,:,:,3] = b4_d[y0:y0+Ts_2,x0:x0+Ts_2]
#    y_train_k[n,:,:,4] = b3_d[y0:y0+Ts_2,x0:x0+Ts_2]
#    y_train_k[n,:,:,5] = b2_d[y0:y0+Ts_2,x0:x0+Ts_2]
#    y_train_k[n,:,:,6] = b8_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    y_train_k[n,:,:,7] = b4_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    y_train_k[n,:,:,8] = b3_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    y_train_k[n,:,:,9] = b2_h[y0:y0+Ts_2,x0:x0+Ts_2]
##                if (n+6)%6 == 0 and n<=len(p_train)-6:
##                    y_train_k[n, 0, :, :] = b6[y0 :y0+Ts_2-r, x0 :x0+ps-r]-b6_r[y0 :y0+ps-r, x0 :x0+ps-r]
##                    y_train_k[n+1, 0, :, :] = b5[y0 :y0+ps-r, x0 :x0+ps-r]-b5_r[y0 :y0+ps-r, x0 :x0+ps-r]
##                    y_train_k[n+2, 0, :, :] = b7[y0 :y0+ps-r, x0 :x0+ps-r]-b7_r[y0 :y0+ps-r, x0 :x0+ps-r]
##                    y_train_k[n+3, 0, :, :] = b8_a[y0 :y0+ps-r, x0 :x0+ps-r]-b8_ar[y0 :y0+ps-r, x0 :x0+ps-r]
##                    y_train_k[n+4, 0, :, :] = b11[y0 :y0+ps-r, x0 :x0+ps-r]-b11_r[y0 :y0+ps-r, x0 :x0+ps-r]
##                    y_train_k[n+5, 0, :, :] = b12[y0 :y0+ps-r, x0 :x0+ps-r]-b12_r[y0 :y0+ps-r, x0 :x0+ps-r]
##                n = n + 1
#    x_train = np.concatenate((x_train, x_train_k))
#    y_train = np.concatenate((y_train, y_train_k))
#    
#    x_val_k = np.ndarray(shape=(len(p_val)-1, Ts_2, Ts_2, N), dtype='float32')
#    y_val_k = np.ndarray(shape=(len(p_val)-1, Ts_2, Ts_2, Out), dtype='float32')
#    n = 0
##            for patch in p_val:
##                y0, x0 = patch[0], patch[1]
#    x_val_k[n,:,:,0] = b5_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_val_k[n,:,:,1] = b6_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_val_k[n,:,:,2] = b7_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_val_k[n,:,:,3] = b8_a_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_val_k[n,:,:,4] = b11_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_val_k[n,:,:,5] = b12_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_val_k[n,:,:,6] = b8_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_val_k[n,:,:,7] = b4_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    x_val_k[n,:,:,8] = b3_h[y0:y0+Ts_2,x0:x0+Ts_2]                
#    x_val_k[n,:,:,9] = b2_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    
##                x_val_k[n,0,:,:] = b5_r[y0:y0+ps,x0:x0+ps]
##                x_val_k[n,1,:,:] = b6_rr[y0:y0+ps,x0:x0+ps]
##                x_val_k[n,2,:,:] = b7_r[y0:y0+ps,x0:x0+ps]
##                x_val_k[n,3,:,:] = b8_ar[y0:y0+ps,x0:x0+ps]
##                x_val_k[n,4,:,:] = b11_r[y0:y0+ps,x0:x0+ps]
##                x_val_k[n,5,:,:] = b12_r[y0:y0+ps,x0:x0+ps]
#
##                x_val_k[n,0,:,:] = b6_r[y0:y0+ps,x0:x0+ps]
##                x_val_k[n,1,:,:] = b8_d[y0:y0+ps,x0:x0+ps]
##                x_val_k[n,2,:,:] = b4_d[y0:y0+ps,x0:x0+ps]
##                x_val_k[n,3,:,:] = b3_d[y0:y0+ps,x0:x0+ps]                
##                x_val_k[n,4,:,:] = b2_d[y0:y0+ps,x0:x0+ps]
#
##                x_val_k[n,0,:,:] = b6_hh[y0:y0+ps,x0:x0+ps]
##                x_val_k[n,1,:,:] = b8_h[y0:y0+ps,x0:x0+ps]
##                x_val_k[n,2,:,:] = b4_h[y0:y0+ps,x0:x0+ps]
##                x_val_k[n,3,:,:] = b3_h[y0:y0+ps,x0:x0+ps]                
##                x_val_k[n,4,:,:] = b2_h[y0:y0+ps,x0:x0+ps]
#
#    
#    y_val_k[n, :, :, 0] = b6[y0:y0+Ts_2,x0:x0+Ts_2]#-b6_r[y0:y0+Ts_2,x0:x0+Ts_2]
#    y_val_k[n, :, :, 1] = b6_r[y0:y0+Ts_2,x0:x0+Ts_2]
#    y_val_k[n,:,:,2] = b8_d[y0:y0+Ts_2,x0:x0+Ts_2]
#    y_val_k[n,:,:,3] = b4_d[y0:y0+Ts_2,x0:x0+Ts_2]
#    y_val_k[n,:,:,4] = b3_d[y0:y0+Ts_2,x0:x0+Ts_2]                
#    y_val_k[n,:,:,5] = b2_d[y0:y0+Ts_2,x0:x0+Ts_2]
#    y_val_k[n,:,:,6] = b8_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    y_val_k[n,:,:,7] = b4_h[y0:y0+Ts_2,x0:x0+Ts_2]
#    y_val_k[n,:,:,8] = b3_h[y0:y0+Ts_2,x0:x0+Ts_2]                
#    y_val_k[n,:,:,9] = b2_h[y0:y0+Ts_2,x0:x0+Ts_2]
##                if (n+6)%6 == 0 and n<=len(p_val)-6:
##                    y_val_k[n, 0, :, :] = b6[y0 :y0+ps-r, x0 :x0+ps-r]-b6_r[y0 :y0+ps-r, x0 :x0+ps-r]
##                    y_val_k[n+1, 0, :, :] = b5[y0 :y0+ps-r, x0 :x0+ps-r]-b5_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
##                    y_val_k[n+2, 0, :, :] = b7[y0+r:y0+ps-r, x0+r:x0+ps-r]-b7_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
##                    y_val_k[n+3, 0, :, :] = b8_a[y0+r:y0+ps-r, x0+r:x0+ps-r]-b8_ar[y0+r:y0+ps-r, x0+r:x0+ps-r]
##                    y_val_k[n+4, 0, :, :] = b11[y0+r:y0+ps-r, x0+r:x0+ps-r]-b11_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
##                    y_val_k[n+5, 0, :, :] = b12[y0+r:y0+ps-r, x0+r:x0+ps-r]-b12_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
##            n = n + 1
#    x_val = np.concatenate((x_val, x_val_k))
#    y_val = np.concatenate((y_val, y_val_k))
#    b11, b8_d, b4_d, b3_d, b2_d, b11_r = None, None, None, None, None, None
#    return x_train, y_train, x_val, y_val, x_test, y_test, x_test2, y_test2
#
#
def load_data_super_resolution2(dataset_folder, patch_side,identity,num_bands):#,indices):    
    #############
    # short names
    path, ps= dataset_folder, patch_side#  , v , indices
    #output_path = output_folder
    #############
    k = identity
    r = 33
    dir_list = os.listdir(path)
    dir_list.sort()
#    dir_list = dir_list[:-2]
#    print(dir_list)
    N = num_bands#-4
    Out = 6+4
    import random
    num = 1
    x_train = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
    y_train = np.ndarray(shape=(0, ps , ps, Out ), dtype='float32')
    x_val = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
    y_val = np.ndarray(shape=(0, ps , ps , Out), dtype='float32')
#    K = 33 # 17# #100 #300
    K1 = 470/2
    K2 = 450/2
    for file in dir_list:
        if file[4:9] == 'B11DR' and file[2]==str(num): #and file[2]<str(7):
            vh_file = file
            vh1_file ='00' + str(num) + '_B08D'+ vh_file[9:]
            vh2_file ='00' + str(num) + '_B04D'+ vh_file[9:]
            vh4_file ='00' + str(num) + '_B03D'+ vh_file[9:]
            vh5_file ='00' + str(num) + '_B02D'+ vh_file[9:]
#            b5 b6 b7 b8a b11 b12  

            vh6_file ='00' + str(num) + '_B'+k+'DR'+ vh_file[9:]
            vh6f_file ='00' + str(num) + '_B'+k+'DH'+ vh_file[9:]
            vh7_file ='00' + str(num) + '_B'+k+ vh_file[9:]
            
            b8_d = imageio.imread(path + vh1_file )       
            
            b4_d = imageio.imread(path + vh2_file )
            b3_d = imageio.imread(path + vh4_file )   
            b2_d = imageio.imread(path + vh5_file ) 
            b6 = imageio.imread(path + vh7_file )
            
            b6_r= imageio.imread(path + vh6_file )
            
            b6_hh= imageio.imread(path + vh6f_file )

            vh1_file ='00' + str(num) + '_B05H1'+ vh_file[9:] # senza H1 per immagine normale
            vh2_file = '00' + str(num) + '_B06H1'+ vh_file[9:]
            vh3_file ='00' + str(num) + '_B07H1'+ vh_file[9:]
            vh4_file ='00' + str(num) + '_B8AH1'+ vh_file[9:]
            vh5_file = '00' + str(num) + '_B11H1'+ vh_file[9:]
            vh6_file ='00' + str(num) + '_B12H1'+ vh_file[9:]

            b5_h1 = imageio.imread(path + vh1_file )
            
            b6_h1 = imageio.imread(path + vh2_file )
            
            b7_h1 = imageio.imread(path+vh3_file )
            
            b8_a_h1 = imageio.imread(path + vh4_file )
#            
            b11_h1 = imageio.imread(path+vh5_file )
            
            b12_h1 = imageio.imread(path + vh6_file )
            
            vh1_file ='00' + str(num) + '_B05DR'+ vh_file[9:] # senza H1 per immagine normale
            vh2_file = '00' + str(num) + '_B06DR'+ vh_file[9:]
            vh3_file ='00' + str(num) + '_B07DR'+ vh_file[9:]
            vh4_file ='00' + str(num) + '_B8ADR'+ vh_file[9:]
            vh5_file = '00' + str(num) + '_B11DR'+ vh_file[9:]
            vh6_file ='00' + str(num) + '_B12DR'+ vh_file[9:]

            b5_r = imageio.imread(path + vh1_file )
            
            b6_rr = imageio.imread(path + vh2_file )
            
            b7_r = imageio.imread(path+vh3_file )
            
            b8_ar = imageio.imread(path + vh4_file )      
#            
            b11_r = imageio.imread(path+vh5_file )
            
            b12_r = imageio.imread(path + vh6_file )

            vh1_file ='00' + str(num) + '_B05DH'+ vh_file[9:] # senza H1 per immagine normale
            vh2_file = '00' + str(num) + '_B06DH'+ vh_file[9:]
            vh3_file ='00' + str(num) + '_B07DH'+ vh_file[9:]
            vh4_file ='00' + str(num) + '_B8ADH'+ vh_file[9:]
            vh5_file = '00' + str(num) + '_B11DH'+ vh_file[9:]
            vh6_file ='00' + str(num) + '_B12DH'+ vh_file[9:]

            b5_h = imageio.imread(path + vh1_file )
            
            b6_h = imageio.imread(path + vh2_file )
            
            b7_h = imageio.imread(path+vh3_file )
            b8_a_h = imageio.imread(path + vh4_file )      
#            
            b11_h = imageio.imread(path+vh5_file )
            b12_h = imageio.imread(path + vh6_file )


            
            vh1_file ='00' + str(num) + '_B02DH'+ vh_file[9:] # senza H1 per immagine normale
            vh2_file = '00' + str(num) + '_B03DH'+ vh_file[9:]
            vh3_file ='00' + str(num) + '_B04DH'+ vh_file[9:]
            vh4_file ='00' + str(num) + '_B08DH'+ vh_file[9:]
            
            b2_h = imageio.imread(path + vh1_file )
            
            b3_h = imageio.imread(path + vh2_file )
            
            b4_h = imageio.imread(path+vh3_file )
            
            b8_h = imageio.imread(path + vh4_file )      

            mask = np.zeros(b6_r.shape,dtype='float32')
            train_patches = np.zeros(b6_r.shape,dtype='float32')
            val_patches = np.zeros(b6_r.shape,dtype='float32')

                
            [s1, s2] = b6_r.shape
            p2 = []
            for y in range(1,s1-ps+1,r):
                for x in range(1,s2-ps+1,r):
                    mask_d0 = mask[y:y+ps,x:x+ps]
                    s_0 =  mask_d0.sum()
                    if s_0 == 0:
                        p2.append([y,x])

#            p = [p2[s] for s in v]                    
            random.shuffle(p2)
            p = p2
#            print(len(p))
#            random.shuffle(p)
            P = int(500)# int(5000)
            p_train,p_val= p[:int(0.8*P)],p[int(0.8*P):P]
            print(len(p_train))
            
            x_train_k = np.ndarray(shape=(len(p_train), ps, ps, N), dtype='float32')
            y_train_k = np.ndarray(shape=(len(p_train), ps , ps, Out), dtype='float32')
            n = 0
            for patch in p_train:
                y0, x0 = patch[0], patch[1]
                x_train_k[n,:,:,0] = b5_h[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,1] = b6_h[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,2] = b7_h[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,3] = b8_a_h[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,4] = b11_h[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,5] = b12_h[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,6] = b8_h[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,7] = b4_h[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,8] = b3_h[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,9] = b2_h[y0:y0+ps,x0:x0+ps]
                
                y_train_k[n, :, :, 0] = b6[y0 :y0+ps , x0 :x0+ps ]#-b6_r[y0 :y0+ps , x0 :x0+ps ]
                y_train_k[n, :, :, 1] = b6_r[y0 :y0+ps , x0 :x0+ps ]
                y_train_k[n,:,:,2] = b8_d[y0 :y0+ps , x0 :x0+ps ]
                y_train_k[n,:,:,3] = b4_d[y0 :y0+ps , x0 :x0+ps ]
                y_train_k[n,:,:,4] = b3_d[y0 :y0+ps , x0 :x0+ps ]
                y_train_k[n,:,:,5] = b2_d[y0 :y0+ps , x0 :x0+ps ]
                y_train_k[n,:,:,6] = b8_h[y0:y0+ps,x0:x0+ps]
                y_train_k[n,:,:,7] = b4_h[y0 :y0+ps , x0 :x0+ps ]
                y_train_k[n,:,:,8] = b3_h[y0 :y0+ps , x0 :x0+ps ]
                y_train_k[n,:,:,9] = b2_h[y0 :y0+ps , x0 :x0+ps ]
                n = n + 1
            x_train = np.concatenate((x_train, x_train_k))
            y_train = np.concatenate((y_train, y_train_k))
            
            x_val_k = np.ndarray(shape=(len(p_val), ps, ps, N), dtype='float32')
            y_val_k = np.ndarray(shape=(len(p_val), ps , ps , Out), dtype='float32')
            n = 0
            for patch in p_val:
                y0, x0 = patch[0], patch[1]
                x_val_k[n,:,:,0] = b5_h[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,1] = b6_h[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,2] = b7_h[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,3] = b8_a_h[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,4] = b11_h[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,5] = b12_h[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,6] = b8_h[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,7] = b4_h[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,8] = b3_h[y0:y0+ps,x0:x0+ps]                
                x_val_k[n,:,:,9] = b2_h[y0:y0+ps,x0:x0+ps]
                

#                y_val_k[n, 0, :, :] = b11[y0 :y0+ps , x0 :x0+ps ]#-b11_r[y0 :y0+ps-r, x0 :x0+ps-r]
                y_val_k[n, :, :, 0] = b6[y0 :y0+ps , x0 :x0+ps ]#-b6_r[y0 :y0+ps , x0 :x0+ps ]
                y_val_k[n, :, :, 1] = b6_r[y0 :y0+ps , x0 :x0+ps ]
                y_val_k[n,:,:,2] = b8_d[y0 :y0+ps , x0 :x0+ps ]
                y_val_k[n,:,:,3] = b4_d[y0 :y0+ps , x0 :x0+ps ]
                y_val_k[n,:,:,4] = b3_d[y0 :y0+ps , x0 :x0+ps ]                
                y_val_k[n,:,:,5] = b2_d[y0 :y0+ps , x0 :x0+ps ]
                y_val_k[n,:,:,6] = b8_h[y0:y0+ps,x0:x0+ps]
                y_val_k[n,:,:,7] = b4_h[y0 :y0+ps , x0 :x0+ps ]
                y_val_k[n,:,:,8] = b3_h[y0 :y0+ps , x0 :x0+ps ]                
                y_val_k[n,:,:,9] = b2_h[y0 :y0+ps , x0 :x0+ps ]
                n = n + 1
            x_val = np.concatenate((x_val, x_val_k))
            y_val = np.concatenate((y_val, y_val_k))
            b11, b8_d, b4_d, b3_d, b2_d, b11_r = None, None, None, None, None, None
            num +=1

    return x_train, y_train, x_val, y_val
#
#def combinations_input_super_resolution(x_trai, y_trai, x_va, y_va,x_trai2, y_trai2, x_va2, y_va2, comb,num):
#    N = num
#    
#    x_train = np.ndarray(shape=(x_trai.shape[0], x_trai.shape[1], x_trai.shape[2], N), dtype='float32')
#    y_train = np.ndarray(shape=(y_trai.shape[0], y_trai.shape[1],y_trai.shape[2], y_trai.shape[3]), dtype='float32')
#    x_val = np.ndarray(shape=(x_va.shape[0], x_va.shape[1], x_va.shape[2], N), dtype='float32')
#    y_val = np.ndarray(shape=(y_va.shape[0], y_va.shape[1],y_va.shape[2], y_va.shape[3]), dtype='float32')
#
#    x_train2 = np.ndarray(shape=(x_trai2.shape[0], x_trai2.shape[1], x_trai2.shape[2], N), dtype='float32')
#    y_train2 = np.ndarray(shape=(y_trai2.shape[0], y_trai2.shape[1],y_trai2.shape[2], y_trai.shape[3]), dtype='float32')
#    x_val2 = np.ndarray(shape=(x_va2.shape[0], x_va2.shape[1], x_va2.shape[2], N), dtype='float32')
#    y_val2 = np.ndarray(shape=(y_va2.shape[0], y_va2.shape[1],y_va2.shape[2], y_va2.shape[3]), dtype='float32')
#
#
#    if comb == "VV":
#        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,0], y_trai, x_va[:,:,:,0], y_va,x_trai2[:,:,:,0], y_trai2, x_va2[:,:,:,0], y_va2
#        x_train, x_val,x_train2, x_val2 = np.expand_dims(x_train, 3),  np.expand_dims(x_val, 3),  np.expand_dims(x_train2, 3), np.expand_dims(x_val2, 3)
#    elif comb == "VH":
#        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,1], y_trai, x_va[:,:,:,1], y_va,x_trai2[:,:,:,1], y_trai2, x_va2[:,:,:,1], y_va2
#        x_train, x_val,x_train2, x_val2 = np.expand_dims(x_train, 3),  np.expand_dims(x_val, 3),  np.expand_dims(x_train2, 3), np.expand_dims(x_val2, 3)
#    elif comb == "VVaVH":
#        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,0:2], y_trai, x_va[:,:,:,0:2], y_va,x_trai2[:,:,:,0:2], y_trai2, x_va2[:,:,:,0:2], y_va2
#    elif comb == "VVaVHaRatio":
#        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,0:3], y_trai, x_va[:,:,:,0:3], y_va,x_trai2[:,:,:,0:3], y_trai2, x_va2[:,:,:,0:3], y_va2
#    elif comb == "VVaVHaSum":
#        x_train[:,:,:,0:2], y_train, x_val[:,:,:,0:2], y_val,x_train2[:,:,:,0:2], y_train2, x_val2[:,:,:,0:2], y_val2 = x_trai[:,:,:,0:2], y_trai, x_va[:,:,:,0:2], y_va,x_trai2[:,:,:,0:2], y_trai2, x_va2[:,:,:,0:2], y_va2
#        x_train[:,:,:,2], x_val[:,:,:,2],x_train2[:,:,:,2], x_val2[:,:,:,2]= x_trai[:,:,:,3], x_va[:,:,:,3],x_trai2[:,:,:,3], x_va2[:,:,:,3]
#    elif comb == "VVaVHaDiff":
#        x_train[:,:,:,0:2], y_train, x_val[:,:,:,0:2], y_val,x_train2[:,:,:,0:2], y_train2, x_val2[:,:,:,0:2], y_val2 = x_trai[:,:,:,0:2], y_trai, x_va[:,:,:,0:2], y_va,x_trai2[:,:,:,0:2], y_trai2, x_va2[:,:,:,0:2], y_va2
#        x_train[:,:,:,2], x_val[:,:,:,2], x_train2[:,:,:,2], x_val2[:,:,:,2] = x_trai[:,:,:,4], x_va[:,:,:,4], x_trai2[:,:,:,4], x_va2[:,:,:,4]
#    elif comb == "Ratio":
#        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai[:,:,:,2], y_trai, x_va[:,:,:,2], y_va,x_trai2[:,:,:,2], y_trai2, x_va2[:,:,:,2], y_va2
#        x_train, x_val,x_train2, x_val2 = np.expand_dims(x_train, 3),  np.expand_dims(x_val, 3),  np.expand_dims(x_train2, 3), np.expand_dims(x_val2, 3)
#    elif comb == "Total":
#        x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = x_trai, y_trai, x_va, y_va,x_trai2, y_trai2, x_va2, y_va2
#    
#    return x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2
#
#
#    folder_1, size,num_bands,indices

def load_data_tandem_x(dataset_folder, patch_side,num_bands,indices):
    
    #############
    # short names
    path, ps, N, v = dataset_folder, patch_side,num_bands, indices
#    ps = 50
#    ps1 = 1000
    ps1 = ps 
    #############
    dir_list = os.listdir(path)
    dir_list.sort()
    print(dir_list)
#    N = 5
    N = N #+ 1
    Out = 2#1#3
    import random
    num = 1
    
    
#    x_train = np.ndarray(shape=(0, ps//4,4*ps, N), dtype='float32')
#    y_train = np.ndarray(shape=(0, ps//4,4*ps, Out), dtype='float32')
#    x_val = np.ndarray(shape=(0, ps//4,4*ps, N), dtype='float32')
#    y_val = np.ndarray(shape=(0, ps//4,4*ps, Out), dtype='float32')
    x_train = np.ndarray(shape=(0, ps,ps1, N), dtype='float32')
    y_train = np.ndarray(shape=(0, ps,ps1, Out), dtype='float32')
    x_val = np.ndarray(shape=(0, ps,ps1, N), dtype='float32')
    y_val = np.ndarray(shape=(0, ps,ps1, Out), dtype='float32')
    
#    "D:\Works\DLR\Uganda_Solberg\uganda_XSAR_noerr3.tif"
    r = 128
            
#    nasa = imageio.imread(path + "line_uganda_NASADEM.tif")
#    "D:\Works\DLR\Uganda_Solberg\uganda_NASADEM_DeNoise_daruotare_fatto_FFT.tif"
    images = [0,1,3,4,5]
    coordinates = [(0,0), (1,0),(0,1),(1,1)]
    for k in range(4): 
        [x_1, y_1] = coordinates[k]
        for z in range(4):
            for x in range(4):
                nasa = imageio.imread(path +"brasil_nasadem_"+str(x_1 )+"_"+str(y_1 )+"_"+str(z )+"_"+str(x )+".tif")#  "uganda_NASADEM_DeNoise_daruotare_fatto_FFT"+str(nnn+1)+".tif")
                xsar2 = imageio.imread(path + "brasil_xsar_"+str(x_1 )+"_"+str(y_1 )+"_"+str(z )+"_"+str(x )+".tif")#fatto.tif") #fatto.tif")# "uganda_XSAR_DeNoise_daruotare_fatto_FFT"+str(nnn+1)+".tif")# "uganda_XSAR_daruotare"+str(nnn+1)+"_fatto.tif")# 
                land = imageio.imread(path + "brasil_landsat_treecover_"+str(x_1 )+"_"+str(y_1 )+"_"+str(z )+"_"+str(x )+".tif")
            
            #        nasa = imageio.imread(path +"uganda_NASADEM_daruotare"+str(nnn+1)+"_fatto.tif")#  "uganda_NASADEM_DeNoise_daruotare_fatto_FFT"+str(nnn+1)+".tif")
            #        xsar3 = imageio.imread(path + "uganda_XSAR_daruotare"+str(nnn+1)+"_fatto.tif")#fatto.tif") #fatto.tif")# "uganda_XSAR_DeNoise_daruotare_fatto_FFT"+str(nnn+1)+".tif")# "uganda_XSAR_daruotare"+str(nnn+1)+"_fatto.tif")# 
            #        land = imageio.imread(path + "uganda_landcover_daruotare"+str(nnn +1)+"_fatto.tif")
            #
            
            
            #        land1 = land == np.min(land)
                diffxtx = nasa - xsar2# xsar3 - nasa 
                diff_land_zero = diffxtx*(land == 0)
                meanx = np.sum(diff_land_zero)/(np.sum(land == 0) + 10**(-10))
                
                nasa = (nasa - meanx)
            #    xsar2 = gaussian_filter(xsar4, (3,3))
            #        diff = (nasa - xsar2)*land1
            #        print(np.sum(diff)/np.count_nonzero(land1))
                print(np.max(land))
                print(np.min(land))
            #        xsar2 = imageio.imread(path + "line_uganda_XSAR_noer.tif")#noerr3.tif")
            #        xsar2 = imageio.imread(path + "line_uganda_XSAR_noerr3_minus_bias.tif")
            #        xsar2 = imageio.imread(path + "line_uganda_XSAR_minus_bias.tif")
            #        xsar2 = imageio.imread(path + "line_uganda_XSAR_minus_bias_noerr3.tif")
            #        xsar2 = imageio.imread(path + "line_uganda_XSAR.tif")
            #        xsar2 = xsar2/20000
            #        nasa = nasa/20000
            #        land = land == np.min(land)
                mask = xsar2 == 0
            #### other additional input    
                [s1, s2] = nasa.shape
                print(nasa.shape)
                p2 = []
            #    for y in range(1,s1-ps//4+1,r): 
            #        for x in range(1,s2-4*ps+1,r):
            #            mask_d0 = mask[y:y+ps//4,x:x+4*ps]
                for y in range(1,s1-ps+1,r): 
                    for x in range(1,s2-ps1+1,r):
                        mask_d0 = mask[y:y+ps,x:x+ps1]
            #                land_0 = land[y:y+ps,x:x+ps]
                        [m1,m2] = mask_d0.shape
                        s_0 =  mask_d0.sum()
            #                s_1 =  land_0.sum()
                        if s_0 == 0 and len(p2) < 5000:# and s_1 == 0:
                            p2.append([y,x])
            #    p = p2[:len(p2):10]#[p2[s] for s in v]                    
            #            random.shuffle(p)
                p = p2
                random.shuffle(p)
                print(len(p2))
                P = len(p) # 10500#7000#len(p2)#int(8000)
                p_train,p_val= p[:int(0.8*P)],p[int(0.8*P):P]
                print(len(p_train))
                print(len(p_val))
                x_train_k = np.ndarray(shape=(len(p_train), ps,ps1, N), dtype='float32')
                y_train_k = np.ndarray(shape=(len(p_train), ps,ps1, Out), dtype='float32')
            #    x_train_k = np.ndarray(shape=(len(p_train), ps//4,4*ps, N), dtype='float32')
            #    y_train_k = np.ndarray(shape=(len(p_train), ps//4,4*ps, Out), dtype='float32')
                n = 0
                for patch in p_train:
                    y0, x0 = patch[0], patch[1]
                    x_train_k[n,:,:,0] = nasa[y0:y0+ps,x0:x0+ps1]
            #            x_train_k[n,:,:,1] = land[y0:y0+ps,x0:x0+ps]
            #        x_train_k[n,:,:,0] = nasa[y0:y0+ps//4,x0:x0+ 4*ps]
            #                tren[y0:y0+ps,x0:x0+ps] += 1
            
                    y_train_k[n,:,:,0]= xsar2[y0:y0+ps,x0:x0+ ps1]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                    y_train_k[n,:,:,1]= xsar2[y0:y0+ps,x0:x0+ ps1] - nasa[y0:y0+ps,x0:x0+ps1] #-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
            #        y_train_k[n,:,:,0]= xsar2[y0:y0+ps//4,x0:x0+ 4*ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                    
                    n = n + 1
                print(y_train_k.shape)
                print(y_train.shape)
                x_train = np.concatenate((x_train, x_train_k))
                y_train = np.concatenate((y_train, y_train_k))
            
            
                x_val_k = np.ndarray(shape=(len(p_val), ps,ps1, N), dtype='float32')
                y_val_k = np.ndarray(shape=(len(p_val), ps,ps1, Out), dtype='float32')
            
            #    x_val_k = np.ndarray(shape=(len(p_val), ps//4, 4*ps, N), dtype='float32')
            #    y_val_k = np.ndarray(shape=(len(p_val), ps//4, 4*ps, Out), dtype='float32')
                
                n = 0
                for patch in p_val:
                    y0, x0 = patch[0], patch[1]
            #        x_val_k[n,:,:,0] = nasa[y0:y0+ps//4,x0:x0+ 4*ps]
                    x_val_k[n,:,:,0] = nasa[y0:y0+ps,x0:x0+ps1]
            #            x_val_k[n,:,:,1] = land[y0:y0+ps,x0:x0+ps]
            #                vald[y0:y0+ps,x0:x0+ps] += 1
            #        y_val_k[n,:,:,0] = xsar2[y0:y0+ps//4,x0:x0+ 4*ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                    y_val_k[n,:,:,0] = xsar2[y0:y0+ps,x0:x0+ ps1]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                    y_val_k[n,:,:,1] = xsar2[y0:y0+ps,x0:x0+ ps1] - nasa[y0:y0+ps,x0:x0+ps1]
                    n = n + 1
                x_val = np.concatenate((x_val, x_val_k))
                y_val = np.concatenate((y_val, y_val_k))
                vv, park, veg, wat, soil= None, None,None,None,None 
                num +=1
                dim1 = xsar2.shape[0]
                dim21 =  xsar2.shape[1] # 
                dim2 = 1000#
            #        x_trainx = np.ndarray(shape=(0, dim1, dim2, N), dtype='float32')
            #        y_trainx = np.ndarray(shape=(0, dim1, dim2, Out), dtype='float32')
            #        x_valx = np.ndarray(shape=(0, dim1, dim2, N), dtype='float32')
            #        y_valx = np.ndarray(shape=(0, dim1, dim2, Out), dtype='float32')
            #
            #        x_train_kx = np.ndarray(shape=(1, dim1, dim2, N), dtype='float32')
            #        y_train_kx = np.ndarray(shape=(1, dim1, dim2, Out), dtype='float32')
            #        x_val_kx = np.ndarray(shape=(1, dim1, dim2, N), dtype='float32')
            #        y_val_kx = np.ndarray(shape=(1, dim1, dim2, Out), dtype='float32')
            #
            #        for n in range(0,dim21 ,100):
            #            if nasa[:,n:n+1000].shape[1] == 1000:
            #                x_val_kx[n,:,:,0] = nasa[:,n:n+1000]
            #                y_val_kx[n,:,:,0] = xsar2[:,n:n+1000]
            #                y_val_kx[n,:,:,1] = xsar2[:,n:n+1000] - nasa[:,n:n+1000]
            #        
            #                x_train_kx[n,:,:,0] = nasa[:,n:n+1000]
            #                y_train_kx[n,:,:,0] = xsar2[:,n:n+1000]
            #                y_train_kx[n,:,:,1] = xsar2[:,n:n+1000] - nasa[:,n:n+ 1000]
            #    
            #    
            #                x_trainx = np.concatenate((x_trainx, x_train_kx))
            #                y_trainx = np.concatenate((y_trainx, y_train_kx))
            #                x_valx = np.concatenate((x_valx, x_val_kx))
            #                y_valx = np.concatenate((y_valx, y_val_kx))
    
    return x_train, y_train, x_val, y_val #  x_trainx, y_trainx, x_valx, y_valx #     


