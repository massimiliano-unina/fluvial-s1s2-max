# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import keras
import pickle as pkl
from sklearn import metrics

from segmentation_models import Unet,PSPNet,Linknet,FPN, Nestnet, Xnet#,ResNeXt50
from Models import FCN8, FCN32, SegNet, UNet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import cce_jaccard_loss, bce_jaccard_loss, cce_dice_loss, dice_loss,bce_dice_loss
from segmentation_models.metrics import iou_score
from segmentation_models.load_data import load_data,combinations_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,Callback
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Input, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import os
from util import train_generator_2, val_generator_2,train_generator_3, val_generator_3, train_ml_models
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import MaxPooling2D, UpSampling2D, Convolution2D, Input, merge, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model

import time
import numpy as np
from tifffile import imsave
from mat4py import savemat, loadmat
import argparse
import gdal

# def custom_loss(y_true, y_pred):
#              loss1=bce_jaccard_loss(y_true,y_pred)
#              loss2=center_loss(y_true,fc)
#              return loss1+lambda*loss2



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



indices = 0 # indo['A']
# time_callback = TimeHistory()

    
# load your data
## training and validation normale
folder_1 = r"D:\Works\Albufera-SemanticSegmentation\S2_S1\\" #r"D:\Albufera\\"#

## testing a parte 
folder_test = folder_1 #r"D:\Works\Albufera-SemanticSegmentation\Testing\S2_S1\\"
size = 128
size_t = 144 # 128 #
n_epochs = 3
n_batch = 32

comb = [ "VVaVH"] # ["Tri"] #  ["VV", "VH", "VVaVH"] #  ["VV", "VH", "VVaVH", "Tri"] # ["VV", "VH"] # ["Tri"]#["VVaVH"] #"VV","Tri","VVaVH"]#"VH",]# ["TriVVaVH"]#  ["VVaVH"]# , "VVaVHaSum","VVaVHaRatio", "VVaVHaDiff","Total"]#"Ratio",  #["VVaVH"]#Ratio"]# ["VVaVH"]##
# num = [1,1,2]#,3,3,3,5]#1,#[2]# 
num = {"VH": 1, "VV": 1, "VVaVH":2, "Tri": 6}
#class_weight = {0: 50., 1: 1., 2: 1.}
class_weight = [40.0, 1.0, 1.0]
for date in range(5):
    date2 = str(date)
    x_traina, y_traina, x_vala, y_vala,x_train2a, y_train2a, x_val2a, y_val2a = load_data(folder_1, size,size_t,indo,indo1, date)
    print(x_traina.shape)
    print(x_vala.shape)

    np.savez("train_MetriAgriFOR_"+ str(date2) + "Date.npz",x_traina = x_traina, y_traina = y_traina, x_vala = x_vala, y_vala = y_vala,x_train2a = x_train2a, y_train2a = y_train2a, x_val2a = x_val2a, y_val2a = y_val2a) #,  x_gtrain = x_gtrai y_train = y_trai  x_gval = x_gval, y_val = y_val)    
    del x_traina, y_traina, x_vala, y_vala,x_train2a, y_train2a, x_val2a, y_val2a