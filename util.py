# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 09:14:56 2020

@author: massi
"""

import os
import imageio
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import *
from keras.utils import to_categorical,multi_gpu_model
from keras import backend as K
#import gdal
from keras.backend.tensorflow_backend import set_session
from keras.losses import categorical_crossentropy, mean_absolute_error, binary_crossentropy
import json
from matplotlib import pyplot
import time
#import skimage
import imageio 
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# configSess = tf.ConfigProto()    
# configSess.gpu_options.allow_growth = True    
# set_session(tf.Session(config=configSess))

eps = K.epsilon()

#def lin_int(x,x2,rate):
#    s = x.shape()
#    y = np.ndarray(s)
#    y2 = np.ndarray(s)
#    for i in range(s[2]):
#        y[i] = skimage.transform.resize(x[i],1/rate,oredr=1)
#        y2[i] = skimage.transform.resize(x2[i],1/rate,oredr=0)
#    return y,y2

def train_generator_mndwi(train_val_p2,number_train, batch_size,data_folder, size,feature, combinations):

    if combinations == "VV":
        inputs_band = 1
        choosen_band = [1]
    elif combinations == "VH":
        inputs_band = 1
        choosen_band = [4]
    elif combinations == "VVaVH":
        inputs_band = 2
        choosen_band = [1,4]
    # elif combinations == "VVaVH":
    #     inputs_band = 2
    #     choosen_band = [1,4]
    while True:
        # ind =  np.load(os.path.join(data_f2, 'train_ind.npy'))
        
        ind = np.arange(number_train)
        np.random.shuffle(ind)
        if combinations == "Tri": 
            chan = feature 
        else: 
            chan = inputs_band #feature #np.sum(feature)
        # for i in range(0,len(ind),batch_size):
        for i in range(0,train_val_p2,batch_size):
            if i + batch_size < len(ind):
                X = np.ndarray(shape=(batch_size,size,size,chan))
                J = np.ndarray(shape=(size,size,chan))
#                Y = np.ndarray(shape=(batch_size,size,size,1))
                Y = np.ndarray(shape=(batch_size,size,size,3))
                for k in range(0,batch_size):
                    H = np.load(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.npy'))                       
                    # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif')) 
                    if combinations == "VV":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "VH":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "VVaVH":     
                        J[:,:,0] = H[:,:,choosen_band[0]]
                        J[:,:,1] = H[:,:,choosen_band[1]]               
                        X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))
                        # X[k,:,:,1] = np.reshape(H[:,:,choosen_band[1]], newshape=(size,size,chan))
                    else:                    
                        X[k,:,:,:] = np.reshape(H[:,:,:chan], newshape=(size,size,chan))
                    # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.tif'))
                    Y_H = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]) + '.npy'))
                    Y[k,:,:,:] = np.reshape(Y_H[:,:,:3], newshape=(size,size,3))
                    # Y[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.npy'))

                Y = Y.astype('float32')
#                print(np.max(Y))   
                yield X,Y
            
def val_generator_mndwi(train_val_p2, number_val, batch_size,data_folder, size,feature, combinations):
    if combinations == "VV":
        inputs_band = 1
        choosen_band = [1]
    elif combinations == "VH":
        inputs_band = 1
        choosen_band = [4]
    elif combinations == "VVaVH":
        inputs_band = 2
        choosen_band = [1,4]
    # elif combinations == "VVaVH":
    #     inputs_band = 2
    #     choosen_band = [1,4]
    while True:
        # ind =  np.load(os.path.join(data_f2, 'train_ind.npy'))
        
        ind = np.arange(number_val)
        np.random.shuffle(ind)
        if combinations == "Tri": 
            chan = feature 
        else: 
            chan = inputs_band #feature #np.sum(feature)
        # for i in range(0,len(ind),batch_size):
        for i in range(0,train_val_p2,batch_size):
            if i + batch_size < len(ind):
                X = np.ndarray(shape=(batch_size,size,size,chan))
                J = np.ndarray(shape=(size,size,chan))
#                Y = np.ndarray(shape=(batch_size,size,size,1))
                Y = np.ndarray(shape=(batch_size,size,size,2))
                for k in range(0,batch_size):
                    H = np.load(os.path.join(data_folder, 'X_val_' + str(ind[i + k]) + '.npy'))                       
                    # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif')) 
                    if combinations == "VV":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "VH":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "VVaVH":     
                        J[:,:,0] = H[:,:,choosen_band[0]]
                        J[:,:,1] = H[:,:,choosen_band[1]]               
                        X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))
                        # X[k,:,:,1] = np.reshape(H[:,:,choosen_band[1]], newshape=(size,size,chan))
                    else:                    
                        X[k,:,:,:] = np.reshape(H[:,:,:chan], newshape=(size,size,chan))
                    # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.tif'))
                    Y_H = np.load(os.path.join(data_folder, 'Y_val_' + str(ind[i + k]) + '.npy'))
                    Y[k,:,:,:] = np.reshape(Y_H[:,:,3:], newshape=(size,size,3))
                    # Y[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.npy'))

                Y = Y.astype('float32')
#                print(np.max(Y))   
                yield X,Y


def crop_data(folder,save_folder,delay,size,train_perc,stride):#(folder,save_folder,rho,taut,delay,size,train_perc,stride):

    f = folder.split('/')
    model_type = f[-1]
    if model_type == '':
        model_type = f[-2]
    model_type = model_type + '_' + delay + 'days_stride_' + str(stride) + '_'
#
#    if rho:
#        model_type = model_type + 'rhoLT_'
#    if taut:
#        model_type = model_type + 'tau_'
    
    model_type = model_type + 'PS_' + str(size)
    save2 = os.path.join(save_folder,model_type)
    if not os.path.exists(save2):
        os.makedirs(save2)               
    cont1 = 0
    num = 0
    
    if len(delay)==1:
        delay1 = '00' + delay
    else:
        delay1 = '0' + delay

    print('Loading images')
    corr = {}
    
    orbits = os.listdir(folder)
    orbits.sort()
    for orbit in orbits: 
        
        zone_fold = os.path.join(folder,orbit)
        zones = os.listdir(zone_fold)
        zones.sort()
        
        
        for zone in zones[1:]:
            

            cont = 0 
            
#            for pol in pols:    
            dates = os.listdir(os.path.join(zone_fold,zone,'30_INF_vv','gamma0'))
#                gammas.sort()
#                g0 = imageio.imread(os.path.join(zone_fold,zone,'gamma0',gammas[0],'geo_gamma0_dB.tiff'))
#                dims = g0.shape
#                ref = np.zeros((dims[0],dims[1],len(gammas)-1))
#                inputs = np.zeros((dims[0],dims[1],2*len(gammas)-1))
            
            
            for data in dates:
                if cont == 0:
                    gammavv = imageio.imread(os.path.join(zone_fold,zone,'30_INF_vv','gamma0',data,'geo_gamma0_dB.tiff'))
                    gammavv = np.reshape(gammavv,(gammavv.shape[0],gammavv.shape[1],1))
                    gammavh = imageio.imread(os.path.join(zone_fold,zone,'30_INF_vh','gamma0',data,'geo_gamma0_dB.tiff'))
                    gammavh = np.reshape(gammavh,(gammavh.shape[0],gammavh.shape[1],1))
                    cont = cont+1
                else:
                    gammav = imageio.imread(os.path.join(zone_fold,zone,'30_INF_vv','gamma0',data,'geo_gamma0_dB.tiff'))
                    gammav = np.reshape(gammav,(gammav.shape[0],gammav.shape[1],1))
                    gammavv = np.append(gammavv,gammav,axis=2)
                    gammah = imageio.imread(os.path.join(zone_fold,zone,'30_INF_vh','gamma0',data,'geo_gamma0_dB.tiff'))
                    gammah = np.reshape(gammah,(gammah.shape[0],gammah.shape[1],1))
                    gammavh = np.append(gammavh,gammah,axis=2)
            
            cohe_fold1 = os.path.join(zone_fold,zone,'30_INF_vv','coh_temp','012_days')
            cohe_fold2 = os.path.join(zone_fold,zone,'30_INF_vh','coh_temp','012_days')
            dates2 = os.listdir(cohe_fold1) 
            contc = 0
#            print(len(dates2))
            for data2 in dates2:
                if contc == 0:
                    cohevv = imageio.imread(os.path.join(cohe_fold1,data2,'geo_coh_temp.tiff'))
                    cohevv = np.reshape(cohevv,(cohevv.shape[0],cohevv.shape[1],1))
                    cohevh = imageio.imread(os.path.join(cohe_fold2,data2,'geo_coh_temp.tiff'))
                    cohevh = np.reshape(cohevh,(cohevh.shape[0],cohevh.shape[1],1))
                    hoa = imageio.imread(os.path.join(zone_fold,zone,'30_INF_vv','bperp',data2,'geo_bperp.tiff'))
                    hoa = np.reshape(hoa,(hoa.shape[0],hoa.shape[1],1))
                    contc += 1 
                else:
                    cohev = imageio.imread(os.path.join(cohe_fold1,data2,'geo_coh_temp.tiff'))
                    cohev = np.reshape(cohev,(cohev.shape[0],cohev.shape[1],1))
                    cohevv = np.append(cohevv,cohev,axis=2)
                    coheh = imageio.imread(os.path.join(cohe_fold1,data2,'geo_coh_temp.tiff'))
                    coheh = np.reshape(coheh,(coheh.shape[0],coheh.shape[1],1))
                    cohevh = np.append(cohevh,coheh,axis=2)
                    hoa1 = imageio.imread(os.path.join(zone_fold,zone,'30_INF_vv','bperp',data2,'geo_bperp.tiff'))
                    hoa1 = np.reshape(hoa1,(hoa1.shape[0],hoa1.shape[1],1))
                    hoa = np.append(hoa,hoa1,axis=2)
            incid = imageio.imread(os.path.join(zone_fold,zone,'30_INF_vv','theta_inc','geo_localthetainc.tiff'))
                
#            if pol == '30_INF_vv':
#                    "H:\data\Jamanxim2019\0508_month\039_orbit\TS_0\30_INF_vv\bperp\20190503m_20190515s\geo_bperp.tiff"
            
#            inps = np.append(inps,hoa,axis=2)
            
            
            glc = imageio.imread(os.path.join(zone_fold,zone,'30_INF_vv','fromglc','fromglc_4classes.tiff'))
            mask = glc == 0 
            glc = glc==130
            p18 = imageio.imread(os.path.join(zone_fold,zone,'30_INF_vv','prodes','prodes2018_c.tiff'))
            p18 = p18 == 215
            
            p19 = imageio.imread(os.path.join(zone_fold,zone,'30_INF_vv','prodes','prodes2019_c.tiff'))
            p19 = p19 == 215
            
            mask = np.bitwise_or(mask,p19)
            ref = np.bitwise_xor(glc,p18)
            
            ref = np.reshape(ref,(ref.shape[0],ref.shape[1],1))
                    


            dims = cohevv.shape
            # MODIFICARE DA QUI!!!!!
            s1 = dims[0]
            s2 = dims[1]
            s3 = dims[2]
#            print(s3)
#            print(len(dates2))
            num += 1
            print('Cropping image ' + str(num))
            
            for i in range(0,s1 - size + 1,stride):
                for j in range(0,s2 - size + 1,stride):
                    if np.sum(mask[i:i+size,j:j+size])==0:
                        y_k = ref[i:i+size,j:j+size,:]
                        np.save(os.path.join(save2,'Y_' + str(cont1//s3) + '.npy'),y_k)
                        for k in range(s3):
                            ide = {'Orbit': orbit, 'T_S' : zone , 'Cohe_Delay' : delay1, 'Patch_size' : size, 'Row' : str(i), 'Col' : str(j), 'Cohe_date' : dates2[k], 'ref_num' : cont1//s3 }
                            corr[str(cont1)] = ide
                            x_k = incid[i:i+size,j:j+size]
                            x_k = np.reshape(x_k,(size,size,1))
                            x_k = np.append(x_k,gammavv[i:i+size,j:j+size,k:k+1],axis=2)
                            x_k = np.append(x_k,gammavh[i:i+size,j:j+size,k:k+1],axis=2)
                            x_k = np.append(x_k,cohevv[i:i+size,j:j+size,k:k+1],axis=2)
                            x_k = np.append(x_k,cohevh[i:i+size,j:j+size,k:k+1],axis=2)                            
                            x_k = np.append(x_k,gammavv[i:i+size,j:j+size,k+1:k+2],axis=2)
                            x_k = np.append(x_k,gammavh[i:i+size,j:j+size,k+1:k+2],axis=2)
                            x_k = np.append(x_k,hoa[i:i+size,j:j+size,k:k+1],axis=2)
                            np.save(os.path.join(save2,'X_' + str(cont1) + '.npy'),x_k)
                            cont1 += 1
                        
#                        y_k = np.append(ref,hoa,axis=2)
    #                    x_k,y_k = lin_int(x_k1,y_k1,size/4)
                        
                        
                    
                        
         
    with open(os.path.join(save2,'Indices.json'),'w') as json_file:
        json.dump(corr,json_file)

    ind = np.arange(cont1)
    np.random.shuffle(ind)
    train_samp = int(cont1*train_perc/100)
    np.save(os.path.join(save2, 'train_ind.npy'),ind[:train_samp])
    np.save(os.path.join(save2, 'val_ind.npy'),ind[train_samp:])
    


            
def train_generator(train_val_p2, batch_size,data_folder,data_f2, size,feature):
    while True:
        ind =  np.load(os.path.join(data_f2, 'train_ind.npy'))
        np.random.shuffle(ind)
        chan = feature #np.sum(feature)
        # for i in range(0,len(ind),batch_size):
        for i in range(0,train_val_p2,batch_size):
            if i + batch_size < len(ind):
                X = np.ndarray(shape=(batch_size,size,size,chan))
#                Y = np.ndarray(shape=(batch_size,size,size,1))
                Y = np.ndarray(shape=(batch_size,size,size,3))
                for k in range(0,batch_size):
                    H = np.load(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.npy'))                       
                    # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif'))                       
                    X[k,:,:,:] = np.reshape(H[:,:,:chan], newshape=(size,size,chan))
                    # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]) + '.tif'))
                    Y[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]) + '.npy'))

                Y = Y.astype('float32')
#                print(np.max(Y))   
                yield X,Y
            
def val_generator(train_val_p2, batch_size,data_folder,data_f2, size,feature):
    while True:
        ind =  np.load(os.path.join(data_f2, 'val_ind.npy'))
        np.random.shuffle(ind)
        chan = feature# np.sum(feature)
        # for i in range(0,len(ind),batch_size):
        for i in range(0,train_val_p2,batch_size):
            if batch_size + i < len(ind):
                X = np.ndarray(shape=(batch_size,size,size,chan))
#                Y = np.ndarray(shape=(batch_size,size,size,1))
                Y = np.ndarray(shape=(batch_size,size,size,3))
                for k in range(0,batch_size):
                    H = np.load(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.npy'))        
                    # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif'))                       
                    X[k,:,:,:] = np.reshape(H[:,:,:chan], newshape=(size,size,chan))
                    # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]) + '.tif'))
                    Y[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]) + '.npy'))

                Y = Y.astype('float32')
#                print(np.min(Y))   
                yield X,Y

def train_ml_models(data_folder, number_train, combinations, size): 
    if combinations == "VV":
        inputs_band = 1
        choosen_band = [1]
    elif combinations == "VH":
        inputs_band = 1
        choosen_band = [4]
    elif combinations == "VVaVH":
        inputs_band = 2
        choosen_band = [1,4]
    # elif combinations == "VVaVH":
    #     inputs_band = 2
    #     choosen_band = [1,4]
        # ind =  np.load(os.path.join(data_f2, 'train_ind.npy'))
    print(number_train//15)
    ind = np.arange(number_train//15)
    np.random.shuffle(ind)
    if combinations == "Tri": 
        chan = feature 
    else: 
        chan = inputs_band #feature #np.sum(feature)
    # for i in range(0,len(ind),batch_size):
    X1 = []
    Y1 = []
    init = time.time()

    for k in range(0,number_train//15):
        J = np.ndarray(shape=(size,size,chan))

        H = np.load(os.path.join(data_folder, 'X_train_' + str(ind[k]) + '.npy'))                       
        # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif')) 
        if combinations == "VV":                    
            X = np.reshape(H[:,:,choosen_band[0]], newshape=(size*size,chan))
        elif combinations == "VH":                    
            X = np.reshape(H[:,:,choosen_band[0]], newshape=(size*size,chan))
        elif combinations == "VVaVH":     
            J[:,:,0] = H[:,:,choosen_band[0]]
            J[:,:,1] = H[:,:,choosen_band[1]]               
            X = np.reshape(J, newshape=(size*size,chan))
            # X[k,:,:,1] = np.reshape(H[:,:,choosen_band[1]], newshape=(size,size,chan))
        else:                    
            X = np.reshape(H[:,:,:chan], newshape=(size*size,chan))
        # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.tif'))
        Y_H = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[k]) + '.npy'))
        Y = np.reshape(Y_H[:,:,:3], newshape=(size*size,3))

        X1.append(X)
        Y1.append(Y)

    fin = time.time() - init
    print(fin)
    print(len(X1))
    print(len(Y1))
    X2 = np.array(X1)
    print(X2.shape)
    X = X2.reshape((X2.shape[0]*X2.shape[1], X2.shape[2]))
    print(X.shape)
    y = np.array(Y1)
    print(y.shape)

    y2 = y.reshape((y.shape[0]*y.shape[1], y.shape[2]))#
    # y2 = y2.ravel()
    print(y2.shape)
    return X, y2 

def train_generator_2(train_val_p2,number_train, batch_size,data_folder, size,feature, combinations):

    if combinations == "VV":
        inputs_band = 1
        choosen_band = [1]
    elif combinations == "VH":
        inputs_band = 1
        choosen_band = [4]
    elif combinations == "VVaVH":
        inputs_band = 2
        choosen_band = [0,1]#[1,4]
    # elif combinations == "VVaVH":
    #     inputs_band = 2
    #     choosen_band = [1,4]
    while True:
        # ind =  np.load(os.path.join(data_f2, 'train_ind.npy'))
        
        ind = np.arange(number_train)
        np.random.shuffle(ind)
        if combinations == "Tri": 
            chan = feature 
        else: 
            chan = inputs_band #feature #np.sum(feature)
        # for i in range(0,len(ind),batch_size):
        for i in range(0,train_val_p2,batch_size):
            if i + batch_size < len(ind):
                X = np.ndarray(shape=(batch_size,size,size,chan))
                J = np.ndarray(shape=(size,size,chan))
#                Y = np.ndarray(shape=(batch_size,size,size,1))
                Y = np.ndarray(shape=(batch_size,size,size,3))
                for k in range(0,batch_size):
                    H = np.load(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.npy'))                       
                    # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif')) 
                    if combinations == "VV":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "VH":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "VVaVH":     
                        J[:,:,0] = H[:,:,choosen_band[0]]
                        J[:,:,1] = H[:,:,choosen_band[1]]               
                        X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))
                        # X[k,:,:,1] = np.reshape(H[:,:,choosen_band[1]], newshape=(size,size,chan))
                    else:                    
                        X[k,:,:,:] = np.reshape(H[:,:,:chan], newshape=(size,size,chan))
                    # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.tif'))
                    Y_H = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]) + '.npy'))
                    Y[k,:,:,:] = np.reshape(Y_H[:,:,:3], newshape=(size,size,3))
                    # Y[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.npy'))
                X = X.astype('float32')
                Y = Y.astype('float32')
#                print(np.max(Y))   
                yield X,Y
            
def val_generator_2(train_val_p2, number_val, batch_size,data_folder, size,feature, combinations):
    if combinations == "VV":
        inputs_band = 1
        choosen_band = [1]
    elif combinations == "VH":
        inputs_band = 1
        choosen_band = [4]
    elif combinations == "VVaVH":
        inputs_band = 2
        choosen_band = [0,1]# [1,4]
    # elif combinations == "VVaVH":
    #     inputs_band = 2
    #     choosen_band = [1,4]
    while True:
        # ind =  np.load(os.path.join(data_f2, 'train_ind.npy'))
        
        ind = np.arange(number_val)
        np.random.shuffle(ind)
        if combinations == "Tri": 
            chan = feature 
        else: 
            chan = inputs_band #feature #np.sum(feature)
        # for i in range(0,len(ind),batch_size):
        for i in range(0,train_val_p2,batch_size):
            if i + batch_size < len(ind):
                X = np.ndarray(shape=(batch_size,size,size,chan))
                J = np.ndarray(shape=(size,size,chan))
#                Y = np.ndarray(shape=(batch_size,size,size,1))
                Y = np.ndarray(shape=(batch_size,size,size,3))
                for k in range(0,batch_size):
                    H = np.load(os.path.join(data_folder, 'X_val_' + str(ind[i + k]) + '.npy'))  
                                         
                    # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif')) 
                    if combinations == "VV":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "VH":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "VVaVH":     
                        J[:,:,0] = H[:,:,choosen_band[0]]
                        J[:,:,1] = H[:,:,choosen_band[1]]               
                        X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))
                        # X[k,:,:,1] = np.reshape(H[:,:,choosen_band[1]], newshape=(size,size,chan))
                    else:                    
                        X[k,:,:,:] = np.reshape(H[:,:,:chan], newshape=(size,size,chan))
                    # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.tif'))
                    Y_H = np.load(os.path.join(data_folder, 'Y_val_' + str(ind[i + k]) + '.npy'))
                    # print(np.max(Y_H))
                    # print(Y_H.shape)
                    Y[k,:,:,:] = np.reshape(Y_H[:,:,:3], newshape=(size,size,3))
                    # Y[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.npy'))
                X = X.astype('float32')
                Y = Y.astype('float32')
#                print(np.max(Y))  
                # print(Y.shape)
                yield X,Y

def unsupervised_train_generator_2(train_val_p2,number_train, batch_size,data_folder, size,feature, combinations):

    if combinations == "B8":
        inputs_band = 1
        choosen_band = [0]
    elif combinations == "VH":
        inputs_band = 1
        choosen_band = [4]
    elif combinations == "VVaVH":
        inputs_band = 2
        choosen_band = [0,1]#[1,4]
    # elif combinations == "VVaVH":
    #     inputs_band = 2
    #     choosen_band = [1,4]
    while True:
        # ind =  np.load(os.path.join(data_f2, 'train_ind.npy'))
        
        ind = np.arange(number_train)#train_val_p2) #
        np.random.shuffle(ind)
        if combinations == "Tri": 
            chan = 3  #feature # 
        else: 
            chan = inputs_band #feature #np.sum(feature)
        # for i in range(0,len(ind),batch_size):
        for i in range(0,number_train, batch_size): #train_val_p2,batch_size):
            if i + batch_size < len(ind):
                X = np.ndarray(shape=(batch_size,size,size,chan))
                J = np.ndarray(shape=(size,size,chan))
                Y1 = np.ndarray(shape=(batch_size,size,size,3))
                Y = [ np.ndarray(shape=(batch_size,size,size,3)), np.ndarray(shape=(batch_size,size,size,chan))]
                for k in range(0,batch_size):
                    H = np.load(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.npy'))                       
                    # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif')) 
                    if combinations == "B8":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "VH":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "VVaVH":     
                        J[:,:,0] = H[:,:,choosen_band[0]]
                        J[:,:,1] = H[:,:,choosen_band[1]]               
                        X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))
                        # X[k,:,:,1] = np.reshape(H[:,:,choosen_band[1]], newshape=(size,size,chan))
                    else:                    
                        X[k,:,:,:] = np.reshape(H[:,:,:chan], newshape=(size,size,chan))
                    # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.tif'))
                    Y_H = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]) + '.npy'))
                    # Y[0] = np.reshape(Y_H[:,:,:3], newshape=(batch_size,size,size,3)).astype('float32')
                    Y1[k,:,:,:] = np.reshape(Y_H[:,:,:3], newshape=(size,size,3))
                    # Y[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.npy'))
                X = X.astype('float32')
                Y[0] = Y1.astype('float32')
                Y[1] = X.astype('float32')
#                print(np.max(Y))   
                yield X,Y
            
def unsupervised_val_generator_2(train_val_p2, number_val, batch_size,data_folder, size,feature, combinations):
    if combinations == "B8":
        inputs_band = 1
        choosen_band = [0]
    elif combinations == "VH":
        inputs_band = 1
        choosen_band = [4]
    elif combinations == "VVaVH":
        inputs_band = 2
        choosen_band = [0,1]# [1,4]
    # elif combinations == "VVaVH":
    #     inputs_band = 2
    #     choosen_band = [1,4]
    while True:
        # ind =  np.load(os.path.join(data_f2, 'train_ind.npy'))
        
        ind = np.arange(number_val)
        np.random.shuffle(ind)
        if combinations == "Tri": 
            chan = 3# feature 
        else: 
            chan = inputs_band #feature #np.sum(feature)
        # for i in range(0,len(ind),batch_size):
        for i in range(0,number_val, batch_size): #train_val_p2,batch_size):
            if i + batch_size < len(ind):
                X = np.ndarray(shape=(batch_size,size,size,chan))
                J = np.ndarray(shape=(size,size,chan))
                Y1 = np.ndarray(shape=(batch_size,size,size,3))
                Y = [np.ndarray(shape=(batch_size,size,size,3)),np.ndarray(shape=(batch_size,size,size,chan))]
                for k in range(0,batch_size):
                    H = np.load(os.path.join(data_folder, 'X_val_' + str(ind[i + k]) + '.npy'))  
                                         
                    # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif')) 
                    if combinations == "B8":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "VH":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "VVaVH":     
                        J[:,:,0] = H[:,:,choosen_band[0]]
                        J[:,:,1] = H[:,:,choosen_band[1]]               
                        X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))
                        # X[k,:,:,1] = np.reshape(H[:,:,choosen_band[1]], newshape=(size,size,chan))
                    else:                    
                        X[k,:,:,:] = np.reshape(H[:,:,:chan], newshape=(size,size,chan))
                    # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.tif'))
                    Y_H = np.load(os.path.join(data_folder, 'Y_val_' + str(ind[i + k]) + '.npy'))
                    # print(np.max(Y_H))
                    # print(Y_H.shape)
                    Y1[k,:,:,:] = np.reshape(Y_H[:,:,:3], newshape=(size,size,3))
                    # Y[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.npy'))
                X = X.astype('float32')
                Y[0] = Y1.astype('float32')
                Y[1] = X.astype('float32')
#                print(np.max(Y))  
                # print(Y.shape)
                yield X,Y

def unsupervised_train_generator_vv(train_val_p2,number_train, batch_size,data_folder, size,feature, combinations):

    if combinations == "VV":
        inputs_band = 1
        choosen_band = [0]
    elif combinations == "TriVV":
        inputs_band = 3
        choosen_band = [0,2,4]
    elif combinations == "VH":
        inputs_band = 1
        choosen_band = [4]
    elif combinations == "VVaVH":
        inputs_band = 2
        choosen_band = [0,1]#[1,4]
    elif combinations == "Tri":
        inputs_band = 6
        # choosen_band = [1,4]
    while True:
        # ind =  np.load(os.path.join(data_f2, 'train_ind.npy'))
        
        ind = np.arange(number_train)#train_val_p2) #
        np.random.shuffle(ind)
        if combinations == "Tri": 
            chan = 6 #3  #feature # 
        else: 
            chan = inputs_band #feature #np.sum(feature)
        # for i in range(0,len(ind),batch_size):
        for i in range(0,number_train, batch_size): #train_val_p2,batch_size):
            if i + batch_size < len(ind):
                X = np.ndarray(shape=(batch_size,size,size,chan))
                J = np.ndarray(shape=(size,size,chan))
                Y1 = np.ndarray(shape=(batch_size,size,size,3))
                Y = [ np.ndarray(shape=(batch_size,size,size,3)), np.ndarray(shape=(batch_size,size,size,chan)),np.ndarray(shape=(batch_size,size,size,3))]
                for k in range(0,batch_size):
                    H = np.load(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.npy'))                       
                    # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif')) 
                    if combinations == "VV":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "VH":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "TriVV":     
                        J[:,:,0] = H[:,:,choosen_band[0]]
                        J[:,:,1] = H[:,:,choosen_band[1]]   
                        J[:,:,2] = H[:,:,choosen_band[2]]               
                        X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))                        
                    elif combinations == "VVaVH":     
                        J[:,:,0] = H[:,:,choosen_band[0]]
                        J[:,:,1] = H[:,:,choosen_band[1]]  
                        X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))
                        # X[k,:,:,1] = np.reshape(H[:,:,choosen_band[1]], newshape=(size,size,chan))
                    else:                    
                        X[k,:,:,:] = np.reshape(H[:,:,:chan], newshape=(size,size,chan))
                    # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.tif'))
                    Y_H = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]) + '.npy'))
                    # Y[0] = np.reshape(Y_H[:,:,:3], newshape=(batch_size,size,size,3)).astype('float32')
                    Y1[k,:,:,:] = np.reshape(Y_H[:,:,:3], newshape=(size,size,3))
                    # Y[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.npy'))
                X = X.astype('float32')
                Y[0] = Y1.astype('float32')
                Y[1] = X.astype('float32')
                Y[2] = Y1.astype('float32')
#                print(np.max(Y))   
                yield X,Y
            
def unsupervised_val_generator_vv(train_val_p2, number_val, batch_size,data_folder, size,feature, combinations):
    if combinations == "VV":
        inputs_band = 1
        choosen_band = [0]
    elif combinations == "TriVV":
        inputs_band = 3
        choosen_band = [0,2,4]
        
    elif combinations == "VH":
        inputs_band = 1
        choosen_band = [4]
    elif combinations == "VVaVH":
        inputs_band = 2
        choosen_band = [0,1]# [1,4]
    elif combinations == "Tri":
        inputs_band = 6 #2
        # choosen_band = [1,4]
    while True:
        # ind =  np.load(os.path.join(data_f2, 'train_ind.npy'))
        
        ind = np.arange(number_val)
        np.random.shuffle(ind)
        if combinations == "Tri": 
            chan = 6 #
        else: 
            chan = inputs_band #feature #np.sum(feature)
        # for i in range(0,len(ind),batch_size):
        for i in range(0,number_val, batch_size): #train_val_p2,batch_size):
            if i + batch_size < len(ind):
                X = np.ndarray(shape=(batch_size,size,size,chan))
                J = np.ndarray(shape=(size,size,chan))
                Y1 = np.ndarray(shape=(batch_size,size,size,3))
                Y = [np.ndarray(shape=(batch_size,size,size,3)),np.ndarray(shape=(batch_size,size,size,chan)),np.ndarray(shape=(batch_size,size,size,3))]
                for k in range(0,batch_size):
                    H = np.load(os.path.join(data_folder, 'X_val_' + str(ind[i + k]) + '.npy'))  
                                         
                    # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif')) 
                    if combinations == "VV":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "VH":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))

                    elif combinations == "TriVV":     
                        J[:,:,0] = H[:,:,choosen_band[0]]
                        J[:,:,1] = H[:,:,choosen_band[1]]          
                        J[:,:,2] = H[:,:,choosen_band[2]]               
                        X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))
                        
                    elif combinations == "VVaVH":     
                        J[:,:,0] = H[:,:,choosen_band[0]]
                        J[:,:,1] = H[:,:,choosen_band[1]]               
                        X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))
                        # X[k,:,:,1] = np.reshape(H[:,:,choosen_band[1]], newshape=(size,size,chan))
                    else:                    
                        X[k,:,:,:] = np.reshape(H[:,:,:chan], newshape=(size,size,chan))
                    # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.tif'))
                    Y_H = np.load(os.path.join(data_folder, 'Y_val_' + str(ind[i + k]) + '.npy'))
                    # print(np.max(Y_H))
                    # print(Y_H.shape)
                    Y1[k,:,:,:] = np.reshape(Y_H[:,:,:3], newshape=(size,size,3))
                    # Y[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.npy'))
                X = X.astype('float32')
                Y[0] = Y1.astype('float32')
                Y[1] = X.astype('float32')
                Y[2] = Y1.astype('float32')
#                print(np.max(Y))  
                # print(Y.shape)
                yield X,Y


def unsupervised_train_generator_vv2(train_val_p2,number_train, batch_size,data_folder, size,feature, combinations):

    if combinations == "VV":
        inputs_band = 1
        choosen_band = [0]
    elif combinations == "TriVV":
        inputs_band = 3
        choosen_band = [0,2,4]
    elif combinations == "TriVH":
        inputs_band = 3
        choosen_band = [1,3,5]
    elif combinations == "VH":
        inputs_band = 1
        choosen_band = [4]
    elif combinations == "VVaVH":
        inputs_band = 2
        choosen_band = [0,1]#[1,4]
    elif combinations == "Tri":
        inputs_band = 6
    elif combinations == "Tri_one":
        inputs_band = 6 #2
        # choosen_band = [1,4]
    while True:
        # ind =  np.load(os.path.join(data_f2, 'train_ind.npy'))
        
        ind = np.arange(number_train)#train_val_p2) #
        np.random.shuffle(ind)
        if combinations == "Tri": 
            chan = 6 #3  #feature # 
        else: 
            chan = inputs_band #feature #np.sum(feature)
        # for i in range(0,len(ind),batch_size):
        for i in range(0,number_train, batch_size): #train_val_p2,batch_size):
            if i + batch_size < len(ind):
                X = np.ndarray(shape=(batch_size,size,size,chan))
                X2 = np.ndarray(shape=(batch_size,size,size,2))
                if combinations == "Tri_one": 
                    J = np.ndarray(shape=(size,size,2))
                else:
                    J = np.ndarray(shape=(size,size,chan))
                Y1 = np.ndarray(shape=(batch_size,size,size,3))
                Y2 = np.ndarray(shape=(batch_size,size,size,3))
                Y = [ np.ndarray(shape=(batch_size,size,size,3)), np.ndarray(shape=(batch_size,size,size,)),np.ndarray(shape=(batch_size,size,size,3))]
                for k in range(0,batch_size):
                    H = np.load(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.npy'))                       
                    # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif')) 
                    if combinations == "VV":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "VH":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "TriVV":     
                        J[:,:,0] = H[:,:,choosen_band[0]]
                        J[:,:,1] = H[:,:,choosen_band[1]]   
                        J[:,:,2] = H[:,:,choosen_band[2]]               
                        X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))                        
                    elif combinations == "VVaVH":     
                        J[:,:,0] = H[:,:,choosen_band[0]]
                        J[:,:,1] = H[:,:,choosen_band[1]]  
                        X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))
                        # X[k,:,:,1] = np.reshape(H[:,:,choosen_band[1]], newshape=(size,size,chan))
                    elif combinations == "Tri_one": 
                        J[:,:,0] = H[:,:,1]
                        J[:,:,1] = H[:,:,4] 
                        X2[k,:,:,:] = np.reshape(J, newshape=(size,size,2))                        
                    else:                    
                        X[k,:,:,:] = np.reshape(H[:,:,:chan], newshape=(size,size,chan))
                    # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.tif'))
                    Y_H = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]) + '.npy'))
                    # Y[0] = np.reshape(Y_H[:,:,:3], newshape=(batch_size,size,size,3)).astype('float32')
                    Y1[k,:,:,:] = np.reshape(Y_H[:,:,:3], newshape=(size,size,3))
                    Y2[k,:,:,:] = np.reshape(Y_H[:,:,3:], newshape=(size,size,3))
                    # Y[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.npy'))
                X = X.astype('float32')
                Y[0] = Y1.astype('float32')
                if combinations == "Tri_one": 
                    Y[1] = X2.astype('float32')
                else: 
                    Y[1] = X.astype('float32')
                Y[2] = Y1.astype('float32')
#                print(np.max(Y))   
                yield X,Y
            
def unsupervised_val_generator_vv2(train_val_p2, number_val, batch_size,data_folder, size,feature, combinations):
    if combinations == "VV":
        inputs_band = 1
        choosen_band = [0]
    elif combinations == "TriVV":
        inputs_band = 3
        choosen_band = [0,2,4]
    elif combinations == "TriVH":
        inputs_band = 3
        choosen_band = [1,3,5]
    elif combinations == "VH":
        inputs_band = 1
        choosen_band = [4]
    elif combinations == "VVaVH":
        inputs_band = 2
        choosen_band = [0,1]# [1,4]
    elif combinations == "Tri":
        inputs_band = 6 #2
    elif combinations == "Tri_one":
        inputs_band = 6 #2
        # choosen_band = [1,4]
    while True:
        # ind =  np.load(os.path.join(data_f2, 'train_ind.npy'))
        
        ind = np.arange(number_val)
        np.random.shuffle(ind)
        if combinations == "Tri": 
            chan = 6 #
        else: 
            chan = inputs_band #feature #np.sum(feature)
        # for i in range(0,len(ind),batch_size):
        for i in range(0,number_val, batch_size): #train_val_p2,batch_size):
            if i + batch_size < len(ind):
                X = np.ndarray(shape=(batch_size,size,size,chan))
                X2 = np.ndarray(shape=(batch_size,size,size,2))
                if combinations == "Tri_one": 
                    J = np.ndarray(shape=(size,size,2))
                else:
                    J = np.ndarray(shape=(size,size,chan))
                Y1 = np.ndarray(shape=(batch_size,size,size,3))
                Y2 = np.ndarray(shape=(batch_size,size,size,3))
                Y = [np.ndarray(shape=(batch_size,size,size,3)),np.ndarray(shape=(batch_size,size,size,3)),np.ndarray(shape=(batch_size,size,size,3))]
                for k in range(0,batch_size):
                    H = np.load(os.path.join(data_folder, 'X_val_' + str(ind[i + k]) + '.npy'))  
                                         
                    # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif')) 
                    if combinations == "VV":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "VH":                    
                        X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    elif combinations == "Tri_one": 
                        J[:,:,0] = H[:,:,1]
                        J[:,:,1] = H[:,:,4] 
                        X2[k,:,:,:] = np.reshape(J, newshape=(size,size,2))                        
                    elif combinations == "TriVV":     
                        J[:,:,0] = H[:,:,choosen_band[0]]
                        J[:,:,1] = H[:,:,choosen_band[1]]          
                        J[:,:,2] = H[:,:,choosen_band[2]]               
                        X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))
                        
                    elif combinations == "VVaVH":     
                        J[:,:,0] = H[:,:,choosen_band[0]]
                        J[:,:,1] = H[:,:,choosen_band[1]]               
                        X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))
                        # X[k,:,:,1] = np.reshape(H[:,:,choosen_band[1]], newshape=(size,size,chan))
                    else:                    
                        X[k,:,:,:] = np.reshape(H[:,:,:chan], newshape=(size,size,chan))
                    # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.tif'))
                    Y_H = np.load(os.path.join(data_folder, 'Y_val_' + str(ind[i + k]) + '.npy'))
                    # print(np.max(Y_H))
                    # print(Y_H.shape)
                    Y1[k,:,:,:] = np.reshape(Y_H[:,:,:3], newshape=(size,size,3))
                    Y2[k,:,:,:] = np.reshape(Y_H[:,:,3:], newshape=(size,size,3))
                    # Y[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.npy'))
                X = X.astype('float32')
                Y[0] = Y1.astype('float32')
                if combinations == "Tri_one": 
                    Y[1] = X2.astype('float32')
                else: 
                    Y[1] = X.astype('float32')
                Y[2] = Y1.astype('float32')
#                print(np.max(Y))  
                # print(Y.shape)
                yield X,Y






def train_generator_3(train_val_p2,number_train, batch_size,data_folder, size,feature, combinations, x_train, y_train):

    if combinations == "VV":
        inputs_band = 1
        choosen_band = [1]
    elif combinations == "VH":
        inputs_band = 1
        choosen_band = [4]
    elif combinations == "VVaVH":
        inputs_band = 2
        choosen_band = [1,4]
    # elif combinations == "VVaVH":
    #     inputs_band = 2
    #     choosen_band = [1,4]
    while True:
        # ind =  np.load(os.path.join(data_f2, 'train_ind.npy'))
        number_train = x_train.shape[0]
        ind = np.arange(number_train)
        np.random.shuffle(ind)
        if combinations == "Tri": 
            chan = feature 
        else: 
            chan = inputs_band #feature #np.sum(feature)
        # for i in range(0,len(ind),batch_size):
        for i in range(0,train_val_p2,batch_size):
            if i + batch_size < len(ind):
                X = np.ndarray(shape=(batch_size,size,size,chan))
                J = np.ndarray(shape=(size,size,chan))
#                Y = np.ndarray(shape=(batch_size,size,size,1))
                Y = np.ndarray(shape=(batch_size,size,size,3))
                for k in range(0,batch_size):
                    # H = np.load(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.npy'))                       
                    # # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif')) 
                    # if combinations == "VV":                    
                    #     X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    # elif combinations == "VH":                    
                    #     X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    # elif combinations == "VVaVH":     
                    #     J[:,:,0] = H[:,:,choosen_band[0]]
                    #     J[:,:,1] = H[:,:,choosen_band[1]]               
                    #     X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))
                    #     # X[k,:,:,1] = np.reshape(H[:,:,choosen_band[1]], newshape=(size,size,chan))
                    # else:                    
                    #     X[k,:,:,:] = np.reshape(H[:,:,:chan], newshape=(size,size,chan))
                    # # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.tif'))
                    # Y_H = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]) + '.npy'))
                    # Y[k,:,:,:] = np.reshape(Y_H[:,:,:3], newshape=(size,size,3))
                    # # Y[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.npy'))
                    X[k,:,:,:] = x_train[i + k, : , : , :]
                    Y[k,:,:,:] = y_train[i + k, : , : , :]
                Y = Y.astype('float32')
#                print(np.max(Y))   
                yield X,Y
            
def val_generator_3(train_val_p2, number_val, batch_size,data_folder, size,feature, combinations, x_val, y_val):
    if combinations == "VV":
        inputs_band = 1
        choosen_band = [1]
    elif combinations == "VH":
        inputs_band = 1
        choosen_band = [4]
    elif combinations == "VVaVH":
        inputs_band = 2
        choosen_band = [1,4]
    # elif combinations == "VVaVH":
    #     inputs_band = 2
    #     choosen_band = [1,4]
    while True:
        # ind =  np.load(os.path.join(data_f2, 'train_ind.npy'))
        number_val = x_val.shape[0]
        ind = np.arange(number_val)
        np.random.shuffle(ind)
        if combinations == "Tri": 
            chan = feature 
        else: 
            chan = inputs_band #feature #np.sum(feature)
        # for i in range(0,len(ind),batch_size):
        for i in range(0,train_val_p2,batch_size):
            if i + batch_size < len(ind):
                X = np.ndarray(shape=(batch_size,size,size,chan))
                J = np.ndarray(shape=(size,size,chan))
#                Y = np.ndarray(shape=(batch_size,size,size,1))
                Y = np.ndarray(shape=(batch_size,size,size,3))
                for k in range(0,batch_size):
                    # H = np.load(os.path.join(data_folder, 'X_val_' + str(ind[i + k]) + '.npy'))                       
                    # # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif')) 
                    # if combinations == "VV":                    
                    #     X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    # elif combinations == "VH":                    
                    #     X[k,:,:,:] = np.reshape(H[:,:,choosen_band[0]], newshape=(size,size,chan))
                    # elif combinations == "VVaVH":     
                    #     J[:,:,0] = H[:,:,choosen_band[0]]
                    #     J[:,:,1] = H[:,:,choosen_band[1]]               
                    #     X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))
                    #     # X[k,:,:,1] = np.reshape(H[:,:,choosen_band[1]], newshape=(size,size,chan))
                    # else:                    
                    #     X[k,:,:,:] = np.reshape(H[:,:,:chan], newshape=(size,size,chan))
                    # # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.tif'))
                    # Y_H = np.load(os.path.join(data_folder, 'Y_val_' + str(ind[i + k]) + '.npy'))
                    # Y[k,:,:,:] = np.reshape(Y_H[:,:,:3], newshape=(size,size,3))
                    # # Y[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.npy'))
                    X[k,:,:,:] = x_val[i + k, : , : , :]
                    Y[k,:,:,:] = y_val[i + k, : , : , :]
                Y = Y.astype('float32')
#                print(np.max(Y))   
                yield X,Y



def build_model1(input_shape):
    inputs3 = Input(shape = input_shape)

    inputs = BatchNormalization(epsilon = 1e-4, axis = -1)(inputs3)  # epsilon is added to the
                                                          # variance in order to avoid 
                                                          # division for 0
    """ ENCODER"""
    
    """ First convolutional level """
    conv1_1 = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(inputs)
    conv1_2 = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(conv1_1)
                        
    pool1 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid',  # Valid Padding is for avoid odd dimension 
                         data_format = 'channels_last')(conv1_2)            # and pads the input with -inf
                         
    
    """ Second convolutional level """
    
    conv2_1 = Conv2D(128, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(pool1)
                        
    conv2_2 = Conv2D(128, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(conv2_1)
    
                        
    pool2 = MaxPooling2D(pool_size = 2, strides=None, padding = 'valid',  
                         data_format = 'channels_last')(conv2_2)         
    
    """ Third convolutional level """
    
    conv3_1 = Conv2D(256, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(pool2)
    
    conv3_2 = Conv2D(256, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(conv3_1)
                        
    pool3 = MaxPooling2D(pool_size = 2, strides = None, padding = 'valid',  
                         data_format = 'channels_last')(conv3_2)     
                         
    """ Last convolutional level """                     
                         
    conv4_1 = Conv2D(512, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(pool3)
    
    conv4_2 = Conv2D(512, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(conv4_1)
                        
                        
    """ DECODER"""
    
    """ Third convolutional level""" 
    
    up3_1 = UpSampling2D(size = (2,2), data_format = "channels_last")(conv4_2)
    
    up3 = Conv2D(256, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(up3_1)
    
    conc3 = Concatenate(axis = -1)([conv3_2,up3])
    
    conv3_1d = Conv2D(256, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(conc3)
    
    conv3_2d = Conv2D(256, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(conv3_1d)
    
    """ Second convolutional level""" 
    
    up2_1 = UpSampling2D(size = (2,2), data_format = "channels_last")(conv3_2d)
    
    up2 = Conv2D(128, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(up2_1)
    
    conc2 = Concatenate(axis = -1)([conv2_2,up2])
    
    conv2_1d = Conv2D(128, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(conc2)
    
    conv2_2d = Conv2D(128, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(conv2_1d)
                        
#    """ Dual Branch """
#    
#    conv1 = Conv2D(128, (3,3), 
#                        padding = 'same', activation = 'relu',
#                        kernel_initializer = 'he_normal',data_format = 'channels_last')(conv2_2d)
#    
#    conv2 = Conv2D(128, (3,3), 
#                        padding = 'same', activation = 'relu',
#                        kernel_initializer = 'he_normal',data_format = 'channels_last')(conv1)
    
    #out12 = Conv2D(1, (1,1), 
    #                    padding = 'same', activation = 'sigmoid',
    #                    kernel_initializer = 'he_normal', name = 'city')(conv2)
                        
    """ First convolutional level""" 
    
    up1_1 = UpSampling2D(size = (2,2), data_format = "channels_last")(conv2_2d)
    
    up1 = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(up1_1)
    
    conc1 = Concatenate(axis = -1)([conv1_2,up1])
    
    conv1_1d = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(conc1)
    
    conv1_2d = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal',data_format = 'channels_last')(conv1_1d)
    
    
    """ Final Layer"""
    
    change = Conv2D(1, (1, 1), 
                        padding = 'same', activation = 'sigmoid',
                        kernel_initializer = 'he_normal',name = 'change',data_format = 'channels_last')(conv1_2d)


    """ Links input and outputs together"""
    #model = Model(inputs = inputs1, outputs = [out6_1,out6_2,out12])
#    model = Model(inputs = [inputs1,inputs2], outputs = out)
    model = Model(inputs = inputs3, outputs = change)
#    model = Model(inputs = inputs3, outputs = [change,change_class])
#    model = multi_gpu_model(model, gpus=2)
    return model


def build_model2(input_shape):

    """ Input layer and batch normalization"""
    inputs1 = Input(shape = input_shape)
    inputs = BatchNormalization(epsilon = 1e-4)(inputs1)  # epsilon is added to the
                                                          # variance in order to avoid 
                                                          # division for 0

    conv1 = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal')(inputs)

    conv1_1 = Concatenate(axis = 3)([inputs,conv1])       
    batch1 = BatchNormalization(epsilon = 1e-4)(conv1_1)              
    conv2 = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal')(batch1)
                        
    conv2_1 = Concatenate(axis = 3)([conv1_1,conv2])     
    
    batch2 = BatchNormalization(epsilon = 1e-4)(conv2_1)
    
    conv3 = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal')(batch2)
                        
    conv3_1 = Concatenate(axis = 3)([conv2_1,conv3])

    batch3 = BatchNormalization(epsilon = 1e-4)(conv3_1)    
                    
    conv4 = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal')(batch3)
                        
    conv4_1 = Concatenate(axis = 3)([conv3_1,conv4])
                        
    batch4 = BatchNormalization(epsilon = 1e-4)(conv4_1) 
    
    conv5 = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal')(batch4)
    
    conv5_1 = Concatenate(axis = 3)([conv4_1,conv5])
    
    batch5 = BatchNormalization(epsilon = 1e-4)(conv5_1) 
    
    conv6 = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal')(batch5)
     
    out6_1 = Conv2D(4, (1, 1), 
                        padding = 'same', activation = 'sigmoid',
                        kernel_initializer = 'he_normal',name = 'change')(conv6)
    
    
    """ Links input and outputs together"""
    model = Model(inputs = inputs1, outputs = out6_1)
    """ Links input and outputs together"""
    #model = Model(inputs = inputs1, outputs = [out6_1,out6_2,out12])
#    model = Model(inputs = [inputs1,inputs2], outputs = out)
#    model = multi_gpu_model(model, gpus=2)
#    model = Model(inputs = inputs3, outputs = [change,change_class])
    return model

def build_model3(input_shape):

    """ Input layer and batch normalization"""
    inputs1 = Input(shape = input_shape)
    inputs = BatchNormalization(epsilon = 1e-4)(inputs1)  # epsilon is added to the
                                                          # variance in order to avoid 
                                                          # division for 0

    conv1 = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal')(inputs)

#    conv1_1 = Concatenate(axis = 3)([inputs,conv1])       
    batch1 = BatchNormalization(epsilon = 1e-4)(conv1)              
    conv2 = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal')(batch1)
                        
    conv2_1 = Add()([batch1,conv2])     
    
    batch2 = BatchNormalization(epsilon = 1e-4)(conv2_1)
    
    conv3 = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal')(batch2)
                        
    conv3_1 = Add()([batch2,conv3])

    batch3 = BatchNormalization(epsilon = 1e-4)(conv3_1)    
                    
    conv4 = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal')(batch3)
                        
    conv4_1 = Add()([batch3,conv4])
                        
    batch4 = BatchNormalization(epsilon = 1e-4)(conv4_1) 
    
    conv5 = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal')(batch4)
    
    conv5_1 = Add()([batch4,conv5])
    
    batch5 = BatchNormalization(epsilon = 1e-4)(conv5_1) 
    
    conv6 = Conv2D(64, (3,3), 
                        padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal')(batch5)
    
    conv6_1 = Add()([batch5,conv6])
    
    batch6 = BatchNormalization(epsilon = 1e-4)(conv6_1)
    
    out6_1 = Conv2D(1, (1, 1), 
                        padding = 'same', activation = 'sigmoid',
                        kernel_initializer = 'he_normal',name = 'change')(batch6)
    
    
    """ Links input and outputs together"""
    model = Model(inputs = inputs1, outputs = out6_1)
    """ Links input and outputs together"""
    #model = Model(inputs = inputs1, outputs = [out6_1,out6_2,out12])
#    model = Model(inputs = [inputs1,inputs2], outputs = out)
#    model = multi_gpu_model(model, gpus=2)
#    model = Model(inputs = inputs3, outputs = [change,change_class])
    return model    

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # References
    Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
    What is a good evaluation measure for semantic segmentation?.
    IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.
    https://en.wikipedia.org/wiki/Jaccard_index
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth + eps) / (sum_ - intersection + smooth + eps)
    return (1 - jac) * smooth
    
def jaccard_loss(y_true, y_pred):
    """
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # References
    Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
    What is a good evaluation measure for semantic segmentation?.
    IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.
    https://en.wikipedia.org/wiki/Jaccard_index
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection+eps) / (sum_ - intersection + eps)
    return (1 - jac)
    
#def bce_jaccard_loss(y_true, y_pred):
#    return 10*jaccard_distance_loss(y_true, y_pred) + 100*categorical_crossentropy(y_true, y_pred) + mean_absolute_error(y_true, y_pred)                
def loss3(y_true, y_pred):
    return jaccard_loss(y_true, y_pred) + binary_crossentropy(y_true, y_pred) + mean_absolute_error(y_true, y_pred)           
##
#def loss3(y_true, y_pred):
#    return categorical_crossentropy(y_true, y_pred)



def L1(y_true,y_pred):
    l1 = K.mean(K.abs(y_true-y_pred))
    num = K.sum(y_true)
    tot = K.cast(K.prod(K.shape(y_true)),'float32')
    return l1*(tot/num)
#    return l1/num

def CCE(y_true,y_pred):
    cce = categorical_crossentropy(y_true,y_pred)
    num = K.sum(y_true)
    tot = K.cast(K.prod(K.shape(y_true)),'float32')
    return cce*(tot/num)
#    return cce/num


def JACC(y_true,y_pred,smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)    
    num = K.sum(y_true)
    tot = K.cast(K.prod(K.shape(y_true)),'float32')
    return (1-jac)*(tot/num)*smooth
#    return (1-jac)/num*smooth

def loss2(y_true,y_pred):
    return JACC(y_true,y_pred) + CCE(y_true,y_pred) + L1(y_true,y_pred)
               
def load_images(data_folder):
    ind =  np.load(os.path.join(data_folder, 'val_ind.npy'))
    app = np.load(os.path.join(data_folder, 'X_' + str(ind[0]) + '.npy'))
    app2 = np.load(os.path.join(data_folder, 'Y_' + str(ind[0]) + '.npy'))
    app2 = app2.astype('bool')
    X = np.ndarray(shape = (len(ind),app.shape[0],app.shape[1],app.shape[2]))
    ref = np.ndarray(shape = (len(ind),app2.shape[0],app2.shape[1],4))
    X[0,:,:,:] = app
    ref[0,:,:,:] = app2#np.sum(app2,axis = -1, dtype='bool')
    for i in range(1,len(ind)):       
        X[i,:,:,:] = np.load(os.path.join(data_folder, 'X_' + str(ind[i]) + '.npy'))
        temp = np.load(os.path.join(data_folder, 'Y_' + str(ind[i]) + '.npy'))
#        temp = temp.astype('bool')
#        ref[i,:,:,0] = np.sum(temp,axis = -1,dtype='bool')
        ref[i,:,:,:] = temp#[:,:,1:]
    ref = ref.astype('float32')
    return X,ref#[X[:,:,:,0:1],X[:,:,:,1:]],ref


def save_images(X,save_folder,data_folder):  
    ind =  np.load(os.path.join(data_folder, 'val_ind.npy'))       
    X = X*255
    X = X.astype('float32')
    for k in range(0,X.shape[0]): 
        for i in range(0,X.shape[-1]):
            np.save(os.path.join(save_folder, 'Yp_' + str(ind[k]) + '_D' + str(i) +'.npy'),X[k,:,:,i])
            imageio.imwrite(os.path.join(save_folder, 'Yp_' + str(ind[k]) + '_D' + str(i) + '.tiff'),X[k,:,:,i])
            Y = np.load(os.path.join(data_folder, 'Y_' + str(ind[k]) + '.npy'))
            Y = Y.astype('bool')
            Y = np.sum(Y,axis=-1)
            Y = Y.astype('float32')
            np.save(os.path.join(save_folder, 'Y_' + str(ind[k]) + '_D' + str(i) +'.npy'),Y)
            imageio.imwrite(os.path.join(save_folder,'Yp_' + str(ind[k]) + '_ref.tiff'),Y)

def save_images2(X,Y,mask,p19,k,save):#,geo,proj):
    if not os.path.exists(save):
        os.makedirs(save)
    X = X#*255
    Y = Y#*255
    mask = mask.astype('float32')
    p19 = p19.astype('float32')
    X = X.astype('float32')
    Y = Y.astype('float32')
#    for i in range(Y.shape[-1]):
    
#    np.save(os.path.join(save,'pred' + str(k) + '.npy'),Y)
    imageio.imwrite(os.path.join(save,'pred' + str(k) + '.tiff'),Y)
#    aa = gdal.Open(os.path.join(save,'pred' + str(k) + '.tiff'),gdal.GA_Update)
#    aa.SetGeoTransform(geo)
#    aa.SetProjection(proj)
##    np.save(os.path.join(save,'ref' + str(k) + '.npy'),X)
    if k==0:
        imageio.imwrite(os.path.join(save,'ref.tiff'),X)
#        aa = gdal.Open(os.path.join(save,'ref.tiff'),gdal.GA_Update)
#        aa.SetGeoTransform(geo)
#        aa.SetProjection(proj)
    #    np.save(os.path.join(save,'mask' + str(k) + '.npy'),mask)
        imageio.imwrite(os.path.join(save,'mask.tiff'),mask)
#        aa = gdal.Open(os.path.join(save,'mask.tiff'),gdal.GA_Update)
#        aa.SetGeoTransform(geo)
#        aa.SetProjection(proj)
    #    np.save(os.path.join(save,'p19' + str(k) + '.npy'),p19)
        imageio.imwrite(os.path.join(save,'p19.tiff'),p19)
#        aa = gdal.Open(os.path.join(save,'p19.tiff'),gdal.GA_Update)
#        aa.SetGeoTransform(geo)
#        aa.SetProjection(proj)


def test_img(path1,model,save,feature):
    dates = os.listdir(os.path.join(path1,'30_INF_vv','gamma0'))
    dates.sort()
    contcc = 0
    for date in dates:
        if contcc == 0 :
            
            gammavv = imageio.imread(os.path.join(path1,'30_INF_vv','gamma0',date,'geo_gamma0_dB.tiff'))
            gammavh = imageio.imread(os.path.join(path1,'30_INF_vh','gamma0',date,'geo_gamma0_dB.tiff'))
            gammavv = np.reshape(gammavv,(gammavv.shape[0],gammavv.shape[1],1))
            gammavh = np.reshape(gammavh,(gammavh.shape[0],gammavh.shape[1],1))
            contcc +=1 
        else:
            gammav = imageio.imread(os.path.join(path1,'30_INF_vv','gamma0',date,'geo_gamma0_dB.tiff'))
            gammav = np.reshape(gammav,(gammav.shape[0],gammav.shape[1],1))
            gammavv = np.append(gammavv,gammav,axis=2)
            gammah = imageio.imread(os.path.join(path1,'30_INF_vh','gamma0',date,'geo_gamma0_dB.tiff'))
            gammah = np.reshape(gammah,(gammah.shape[0],gammah.shape[1],1))
            gammavh = np.append(gammavh,gammah,axis=2)
            
            
    cohe_fold1 = os.path.join(path1,'30_INF_vv','coh_temp','012_days')
    cohe_fold2 = os.path.join(path1,'30_INF_vh','coh_temp','012_days')
    dates2 = os.listdir(cohe_fold1) 
    contc = 0
#            print(len(dates2))
    for data2 in dates2:
        if contc == 0:
            cohevv = imageio.imread(os.path.join(cohe_fold1,data2,'geo_coh_temp.tiff'))
            cohevv = np.reshape(cohevv,(cohevv.shape[0],cohevv.shape[1],1))
            cohevh = imageio.imread(os.path.join(cohe_fold2,data2,'geo_coh_temp.tiff'))
            cohevh = np.reshape(cohevh,(cohevh.shape[0],cohevh.shape[1],1))
            hoa = imageio.imread(os.path.join(path1,'30_INF_vv','bperp',data2,'geo_bperp.tiff'))
            hoa = np.reshape(hoa,(hoa.shape[0],hoa.shape[1],1))
            contc += 1 
        else:
            cohev = imageio.imread(os.path.join(cohe_fold1,data2,'geo_coh_temp.tiff'))
            cohev = np.reshape(cohev,(cohev.shape[0],cohev.shape[1],1))
            cohevv = np.append(cohevv,cohev,axis=2)
            coheh = imageio.imread(os.path.join(cohe_fold1,data2,'geo_coh_temp.tiff'))
            coheh = np.reshape(coheh,(coheh.shape[0],coheh.shape[1],1))
            cohevh = np.append(cohevh,coheh,axis=2)
            hoa1 = imageio.imread(os.path.join(path1,'30_INF_vv','bperp',data2,'geo_bperp.tiff'))
            hoa1 = np.reshape(hoa1,(hoa1.shape[0],hoa1.shape[1],1))
            hoa = np.append(hoa,hoa1,axis=2)
    incid = imageio.imread(os.path.join(path1,'30_INF_vv','theta_inc','geo_localthetainc.tiff'))


#    X = np.ndarray(shape=(1,g0.shape[0],g0.shape[1],2*len(gammas)-1))
#    Y = np.ndarray(shape=(g0.shape[0],g0.shape[1],len(gammas)-1))
    glc = imageio.imread(os.path.join(path1,'30_INF_vv','fromglc','fromglc_4classes.tiff'))
    mask = glc == 0 
    glc = glc == 130
    p18 = imageio.imread(os.path.join(path1,'30_INF_vv','prodes','prodes2018_c.tiff'))
    p18 = p18 == 215
    
    p19 = imageio.imread(os.path.join(path1,'30_INF_vv','prodes','prodes2019_c.tiff'))
    p19 = p19 == 215
    
#    aa = gdal.Open(os.path.join(path1,'30_INF_vv','prodes','prodes2019_c.tiff'))
#    geo = aa.GetGeoTransform()
#    proj = aa.GetProjection()
#        mask = np.bitwise_or(mask,p19)
    ref = np.bitwise_xor(glc,p18)
    for k in range(cohevv.shape[2]):
        x_k = incid
        x_k = np.reshape(x_k,(gammavv.shape[0],gammavv.shape[1],1))
        x_k = np.append(x_k,gammavv[:,:,k:k+1],axis=2)
        x_k = np.append(x_k,gammavh[:,:,k:k+1],axis=2)
        x_k = np.append(x_k,cohevv[:,:,k:k+1],axis=2)
        x_k = np.append(x_k,cohevh[:,:,k:k+1],axis=2)      
        x_k = np.append(x_k,gammavv[:,:,k+1:k+2],axis=2)
        x_k = np.append(x_k,gammavh[:,:,k+1:k+2],axis=2)
        x_k = np.append(x_k,hoa[:,:,k:k+1],axis=2)
        x_k = np.reshape(x_k,(1,x_k.shape[0],x_k.shape[1],x_k.shape[2]))
        
#        x_k = np.append(x_k,hoa[:,:,k:k+1],axis=2)
        x_k = x_k[:,:,:,feature]
        dims = x_k.shape
#        print(inp1.shape)
        res1 = dims[1]%64
        res2 = dims[2]%64
        tx1 = int(res1/2)
        ty1 = int(res2/2)
        tx2 = None
        ty2 = None 
        if res1%2==0 and res1!=0:
            tx2 = -tx1
        else:
            tx2 = tx1+1
            tx2 = - tx2
            
        if res2%2==0 and res2!=0:
            ty2 = - ty1
        else:
            ty2 = ty1+1
            ty2 = - ty2
        x_k = x_k[:,tx1:tx2,ty1:ty2,:]
        Y3 = np.zeros((dims[1],dims[2]))
#        print(x_k.shape)
#        print(int(dims[1]/8))
        K = x_k.shape
        for s in range(0,K[1],int(K[1]/8)):
            for t in range(0,K[2],int(K[2]/8)):

                Y2 = model.predict(x_k[:,s:s+int(K[1]/8),t:t+int(K[2]/8),:])
                Y2 = Y2[0,:,:,0]
                Y2 = Y2>=0.5
                Y3[tx1+s:tx1+s+int(K[1]/8),ty1+t:ty1+t+int(K[2]/8)] = Y2

#        for s in range(0,k1,int(k1/8)):
#            for t in range(0,k2,int(k2/8)):
#                Y4[s:s+int(k1/8),t:+t+int(k2/8)] = Y2[s:s+int(k1/8),t:t+int(k2/8)]
        save_images2(ref,Y3,mask,p19,k,save)#,geo,proj)
#        del Y3,x_k
    
    
def test2(data_folder,model,save_folder):
    orbits = os.listdir(data_folder)
    orbits.sort()
    for orbit in orbits:
        scenes = os.listdir(os.path.join(data_folder,orbit))
        scenes.sort()
        for scene in scenes:
            if not os.path.exists(os.path.join(save_folder,orbit,scene)):
                os.makedirs(os.path.join(save_folder,orbit,scene))
            test_img(os.path.join(data_folder,orbit,scene),model,os.path.join(save_folder,orbit,scene))

    
def metrics(Y,Ypred,output):
#    Ypred2 = to_categorical(Ypred)
#    Y = to_categorical(Y)
    TP = []
    TN = []
    FP = []
    FN = [] 
    f =  open(os.path.join(output,'Perf.txt'),'w') 
    f.write('TP      TN      FP     FN')
    for i in range(Ypred.shape[-1]):
        Yi = Y[:,:,:]#,i]         
        Ypi = Ypred[:,:,:]>=0.5#,i]       
        TP.append((Yi*Ypi).sum())
        TN.append(((1-Yi)*(1-Ypi)).sum())
        FP.append(((1-Yi)*(Ypi)).sum())
        FN.append(((Yi)*(1-Ypi)).sum())
        f.write(str(TP[i]) + '  '  + str(TN[i]) + '  ' + str(FP[i]) + '  ' + str(FN[i]))
    f.close

    return TP,TN,FP,FN   

        
