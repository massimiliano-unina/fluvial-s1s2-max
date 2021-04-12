# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from segmentation_models import Unet,PSPNet,Linknet,FPN
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from segmentation_models.load_data import combinations_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,Callback
from keras.layers import Conv2D, Input, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

import time
import numpy as np
from tifffile import imsave
from mat4py import savemat, loadmat
import argparse
import sys
import os
from tifffile import imsave
import numpy as np
import gdal
from scipy.misc import imresize
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
        if file.lower().find('albufera_vv2.tif') != -1 and file[0]==str(num):# and num < 2:
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
            
            vh_file =  str(num) + '_Albufera_VH2' + vv_file[-4:]
            dataset = gdal.Open(path + vh_file, gdal.GA_ReadOnly)
            vh = dataset.ReadAsArray()
            dataset = None

            dataset = gdal.Open(path + vv_file, gdal.GA_ReadOnly)
            vv = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path + park_file, gdal.GA_ReadOnly)
            park = dataset.ReadAsArray()
            dataset = None
            vv = vv[0,:,:]
            vh = vh[0,:,:]
            veg = veg[0,:,:]/255
            soil = soil[0,:,:]/255
            wat = wat[0,:,:]/255
            park = 1 - park/255
            vv_vh = vv
            ave_vv_vh = vv
            diff_vv_vh =vv
            p_train = []
            p_val = []
            p_val =[[1080,1060], [2250,1250],[1080 + size,1060 + size],[1080 + size,1060], [1080,1060 + size] ]
            print(len(p_val))
            x_val_k = np.ndarray(shape=(len(p_val), ps, ps, N), dtype='float32')
            y_val_k = np.ndarray(shape=(len(p_val), ps, ps, Out), dtype='float32')
            
            n = 0
            for patch in p_val:
                y0, x0 = patch[0], patch[1]
                print(y0 )
                print(x0)
                print(y0 + ps)
                print(x0+ps)
                print(vv[y0:y0+ps,x0:x0+ps].shape)
                print(x_val_k[n,:,:,0].shape)
                x_val_k[n,:,:,0] = vv[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,1] = vh[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,2] = vv_vh[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,3] = ave_vv_vh[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,4] = diff_vv_vh[y0:y0+ps,x0:x0+ps]
#                vald[y0:y0+ps,x0:x0+ps] += 1
                y_val_k[n,:,:,0] = soil[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_val_k[n,:,:,1] = veg[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_val_k[n,:,:,2] = wat[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                n = n + 1
            x_val = np.concatenate((x_val, x_val_k))
            y_val = np.concatenate((y_val, y_val_k))
#            x_val2 = np.concatenate((x_val2, x_val_k2))
#            y_val2 = np.concatenate((y_val2, y_val_k2))
            vv, park, veg, wat, soil= None, None,None,None,None 
            num +=1

    return x_train, y_train, x_val, y_val, x_train2, y_train2, x_val2, y_val2#, tren, vald,


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

#parser = argparse.ArgumentParser(description = "Translate everything!")
#parser.add_argument('--indices', required = True)
#args = parser.parse_args()
#indices = args.indices
indo = loadmat(r"C:\Users\massi\Downloads\segmentation_models-master\docs\indices.mat")
indo1 = loadmat(r"C:\Users\massi\Downloads\segmentation_models-master\docs\indices1.mat")

indices = 0#indo['A']
time_callback = TimeHistory()

    
# load your data
## training and validation normale
folder_1 = r"D:\Works\Albufera-SemanticSegmentation\S2_S1\\" #r"D:\Albufera\\"#

## testing a parte 
#folder_1 = r"D:\Works\Albufera-SemanticSegmentation\Testing\S2_S1\\"
size = 128
size_t = 144 #, 128,128]# 128
n_epochs = 10
n_batch = 32

comb =["VH", "VV","VVaVH", "VVaVHaSum","VVaVHaRatio","VVaVHaDiff","Total"]#"VVaVH"]#Ratio"]# ["VVaVH"]#"Ratio","VH", "VV","VVaVH", "VVaVHaSum","VVaVHaRatio","VVaVHaDiff","Total"]#
num =[1,1,2,3,3,3,5]# [2]#1,#

#class_weight = {0: 50., 1: 1., 2: 1.}
class_weight = [40.0, 1.0, 1.0]

x_traina, y_traina, x_vala, y_vala,x_train2a, y_train2a, x_val2a, y_val2a = load_data(folder_1, size,size_t,indo,indo1)
print(x_traina.shape)
print(x_vala.shape)
for combinations in range(len(comb)):
    x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = combinations_input(x_traina, y_traina, x_vala, y_vala,x_train2a, y_train2a, x_val2a, y_val2a, comb[combinations],num[combinations] )
    N = x_train.shape[-1]
    print(x_train.shape)
    print(x_val.shape)
    backs = ['mobilenetv2']# ['resnet34','mobilenetv2']
    for k_back in range(len(backs)):
        BACKBONE = backs[k_back]# 
        preprocess_input = get_preprocessing(BACKBONE)        ## define model and chose between following models: 
        
        networks = ["Unet","Linknet","FPN"]#,"PSPNet"]#["PSPNet","Linknet","FPN"]# 
        for k_mod in range(len(networks)):
        #    size = size_t[k_mod]
            name_model = networks[k_mod]+ "5"# "_despeck3_"#+"3"# 
            if k_mod == 0:
                if N == 3: 
                    model = Unet(BACKBONE,input_shape=(size,size, 3), classes=3, activation='softmax', encoder_weights='imagenet', freeze_encoder=False)
                else:
                    base_model = Unet(BACKBONE,input_shape=(size,size, 3), classes=3, activation='softmax', encoder_weights='imagenet', freeze_encoder=False)
                    inp = Input(shape=(size, size, N))
                    bn = BatchNormalization()(inp)
                    l1 = Conv2D(3, (1, 1))(bn) # map N channels data to 3 channels
                    out = base_model(l1)
                    model = Model(inp, out, name=base_model.name)
                
#            elif k_mod == 1:
#                N = x_train.shape[-1]
#                if N == 3: 
#                    model = PSPNet(BACKBONE, input_shape=(size_t, size_t, 3), classes=3, activation='softmax', encoder_weights='imagenet', encoder_freeze=False)
#                else:
#                    base_model = PSPNet(BACKBONE, input_shape=(size_t, size_t, 3), classes=3, activation='softmax', encoder_weights='imagenet', encoder_freeze=False)
#                    inp = Input(shape=(size_t, size_t, N))
#                    bn = BatchNormalization()(inp)
#                    l1 = Conv2D(3, (1, 1))(bn) # map N channels data to 3 channels
#                    out = base_model(l1)
#                    model = Model(inp, out, name=base_model.name)
                
            elif k_mod == 1:
                N = x_train.shape[-1]
                if N == 3: 
                    model = Linknet(BACKBONE, input_shape=(size, size, 3), classes=3, activation='softmax', encoder_weights='imagenet',encoder_freeze=False)
                else:
                    base_model = Linknet(BACKBONE, input_shape=(size, size, 3), classes=3, activation='softmax', encoder_weights='imagenet',encoder_freeze=False)
                    inp = Input(shape=(size, size, N))
                    bn = BatchNormalization()(inp)
                    l1 = Conv2D(3, (1, 1))(bn) # map N channels data to 3 channels
                    out = base_model(l1)
                    model = Model(inp, out, name=base_model.name)
                
            else:
                N = x_train.shape[-1]
                if N == 3: 
                    model = FPN(BACKBONE, input_shape=(size, size, 3), classes=3, activation='softmax', encoder_weights='imagenet', encoder_freeze=False)
                else:
                    base_model = FPN(BACKBONE, input_shape=(size, size, 3), classes=3, activation='softmax', encoder_weights='imagenet', encoder_freeze=False)
                    inp = Input(shape=(size, size, N))
                    bn = BatchNormalization()(inp)
                    l1 = Conv2D(3, (1, 1))(bn) # map N channels data to 3 channels
                    out = base_model(l1)
                    model = Model(inp, out, name=base_model.name)
                    
#            Adamax = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#            model.compile(loss=bce_jaccard_loss, metrics=[iou_score], optimizer=Adamax)
######### TRAINING ############
##            model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
##            
#            callbacks = [
#                EarlyStopping(patience=10, verbose=1),
#                ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
#                ModelCheckpoint(comb[combinations] + "_" + BACKBONE + "_" + name_model + "_temporary_model_IoU"+ str(size) +".h5", verbose=1, save_best_only=True, save_weights_only=True),
#                time_callback
#            ]
#            
#            ## fit model
#            if name_model == "PSPNet":
#                model.fit(
#                    x=x_train2,
#                    y=y_train2,
#                    batch_size=n_batch,
#                    epochs=n_epochs,
#                    class_weight = class_weight,
#                    validation_data=(x_val2, y_val2),callbacks = callbacks
#                )
#            else: 
#                model.fit(
#                    x=x_train,
#                    y=y_train,
#                    batch_size=n_batch,
#                    epochs=n_epochs,
#                    class_weight = class_weight,
#                    validation_data=(x_val, y_val),callbacks = callbacks
#                )
#            times = time_callback.times
#            dic_times = {}
#            dic_times['times'] = times
#            savemat(comb[combinations] + "_" + BACKBONE + '_' + name_model + '_times.mat', dic_times)
#            model.save_weights(comb[combinations] + "_" + BACKBONE + "_" + name_model + "_model_wIoU"+ str(size) +".h5")
 ############END TRAINING#############           
            
            # Load best model
#            model.load_weights(comb[combinations] + "_" + BACKBONE + "_" + name_model + "_model_wIoU"+ str(size) +".h5")
#                                comb[combinations] + "_" + BACKBONE + "_" + name_model + "_model_wIoU"+ str(size) +".h5"
        #    model.evaluate(x_val, y_val, verbose=1)
            # Predict on train, val and test
            if name_model == "PSPNet":
                preds_train = model.predict(x_train2, verbose=1)
                preds_val = model.predict(x_val2, verbose=1)
                for k in range(8): 
                    x_val_1 = x_val2[50*k,:,:,:]
                    y_val_1 = y_val2[50*k,:,:,:]
                    pred_val_1 = preds_val[50*k,:,:,:]
                    
                    ndvi = y_val_1.astype('float32')
                    im = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_target_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                    imsave(im, ndvi)
                    
                    ndvi2 = pred_val_1.astype('float32')
                    im2 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_output_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                    imsave(im2, ndvi2)
                    
                    ndvi3 = x_val_1.astype('float32')
                    im3 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_input_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                    imsave(im3, ndvi3)
            else:
#                preds_train = model.predict(x_train, verbose=1)
                preds_val = model.predict(x_val, verbose=1)
                
                for k in range(25): 
                    x_val_1 = x_val[k,:,:,:]
                    y_val_1 = y_val[k,:,:,:]
                    pred_val_1 = preds_val[k,:,:,:]
                    
                    ndvi = y_val_1.astype('float32')
                    im = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_target_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                    imsave(im, ndvi)
                    
                    ndvi2 = pred_val_1.astype('float32')
                    im2 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_output_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                    imsave(im2, ndvi2)
                    if N == 1: 
                        ndvi3 = x_val_1[:,:,0].astype('float32')
                        im3 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                        imsave(im3, ndvi3)
                    else: 
                        ndvi3 = x_val_1[:,:,0].astype('float32')
                        im3 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                        imsave(im3, ndvi3)
                        ndvi4 = x_val_1[:,:,1].astype('float32')
                        im4 =  comb[combinations] + "_" + BACKBONE + '_' + name_model + '_VH_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                        imsave(im4, ndvi4)
    
    
