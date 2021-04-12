# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:57:17 2020

@author: massi
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle as pkl
from sklearn import metrics
import keras
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
from keras.losses import MAE, mean_absolute_error, categorical_crossentropy

import os
from util import train_generator, val_generator, train_generator_2, val_generator_2,train_generator_3, val_generator_3, train_ml_models
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
import tensorflow as tf 
from keras.utils import multi_gpu_model

from keras.backend.tensorflow_backend import set_session
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# configSess = tf.ConfigProto()    
# configSess.gpu_options.allow_growth = True    
# set_session(tf.Session(config=configSess))
# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

indices = 0#indo['A']
time_callback = TimeHistory()

data_folder = r"D:\German_Train\\"   
dir_list = os.listdir(data_folder)
dir_list.sort()

train_val_p = 0
test_p = 0
for file1 in dir_list:
    if file1.find("X_test") == -1: 
        train_val_p += 1
    else:
        test_p += 1
        
print(train_val_p)
print(test_p)

folder_out_2 = r"D:\German_Indices\\"   

if not os.path.exists(folder_out_2):
    os.makedirs(folder_out_2)


# ind = np.arange(train_val_p)
# np.random.shuffle(ind)
# train_perc = 0.9
# train_samp = int(train_val_p*train_perc)
# np.save(os.path.join(folder_out_2, 'train_ind.npy'),ind[:train_samp])
# np.save(os.path.join(folder_out_2, 'val_ind.npy'),ind[train_samp:])


# load your data

## testing a parte 
size = 128
size_t = 144 #, 128,128]# 128
n_epochs = 10 # 1
n_batch = 32 # 4
l_rate = 0.005# 0.02 #0.001
#"case_A",
comb = ["case_all"]# ["case_9"] #["case_3_bNDVI"]#["case_11"]# ["case_10"]# [ "case_3_bis", "case_6_bis", "case_1", "case_2", "case_3_wo_Tau", "case_7_Andrew", "case_3",  "case_4", "case_5","case_7",  "case_8", "case_9"] #  ["case_5_HRL"] #
num =  [6] # [4]#[3]# [3, 5, 2, 3, 2, 3, 4, 5, 6, 3, 4, 5] # [6] #  #
N = 6 #np.max(num)#[0]
dic = {} 
for kk in range(len(comb)):
    dic[comb[kk]] = num[kk]
n_classes = 3    
for combinations in comb: 
    N = dic[combinations]    
    backs = ['mobilenetv2']# ['resnet34','mobilenetv2']
    for k_back in backs:
        BACKBONE = k_back # 
        preprocess_input = get_preprocessing(BACKBONE)        ## define model and chose between following models: 
        
        networks = ["DenseUnet"] # [ "FractalNet","Linknet","SegNet"] # ["shallow_CNN", , "Unet","Linknet","SegNet","FPN"] # ["SegNet","FPN"]#   "NestNet" "shallow_CNN"] #  ["SVM", "RF", "GBC", 
        
        for k_mod in networks:
        #    size = size_t[k_mod]
            name_model = k_mod +"_HRL"#+ "_despeck3_"# +"3"#
            name_model_pre = name_model + "_" + combinations + "_1585067315" #{}".format(int(time.time()))
            name_model = name_model + "_" + combinations + "_post" #{}".format(int(time.time()))
            tensorboard = TensorBoard(log_dir='logs/{}'.format(name_model))
            
            if k_mod == "UNet":
                N = 6
                if N == 3: 
                    model = Unet(BACKBONE,input_shape=(size,size, 3), classes=n_classes, activation='softmax')#, encoder_weights='imagenet', freeze_encoder=False)
                else:
                    
                    base_model = Unet(BACKBONE,input_shape=(size,size, 3), classes=n_classes, activation='softmax', encoder_weights='imagenet', freeze_encoder=False)
                    inp = Input(shape=(size, size, N))
                    bn = BatchNormalization()(inp)
                    l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
                    out = base_model(l1)
                    model = Model(inp, out, name=base_model.name)
#                    model = Unet(BACKBONE,input_shape=(size,size, 4), classes=3, activation='softmax', encoder_weights='imagenet', freeze_encoder=False)
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
                
            elif k_mod == "LinkNet":
                # N = x_train.shape[-1]
                if N == 3: 
                    model = Linknet(BACKBONE, input_shape=(size, size, 3), classes=3, activation='softmax', encoder_weights='imagenet',encoder_freeze=False)
                else:
                    base_model = Linknet(BACKBONE, input_shape=(size, size, 3), classes=3, activation='softmax', encoder_weights='imagenet',encoder_freeze=False)
                    inp = Input(shape=(size, size, N))
                    bn = BatchNormalization()(inp)
                    l1 = Conv2D(3, (1, 1))(bn) # map N channels data to 3 channels
                    out = base_model(l1)
                    model = Model(inp, out, name=base_model.name)
#            elif k_mod == "shallow":
            elif k_mod == "shallow":
#                N = x_train.shape[-1]
#                active = 'relu'
#                active3 = 'softmax'
#                inp1 = Input(shape=(None, None, N-1))
#                l1 = UpSampling2D(size=(5,5), name="up_layer")(inp1)
##                bn = BatchNormalization()(l1)
#                inp2 = Input(shape=(None, None, 1))
#                l4 = concatenate([l1, inp2], axis=-1)
#                bn = BatchNormalization()(l4)
#                l2 = Conv2D(64, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None) kernel_initializer='he_normal')(bn)
#                l3 = Conv2D(48, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None) kernel_initializer='he_normal')(l2)
#                l4 = Conv2D(32, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None) kernel_initializer='he_normal')(l3)
#
#                out = Conv2D(3, kernel_size=3, activation=active3, padding='same', kernel_initializer='he_normal')(l4)

                # N = x_train.shape[-1]
                # N = 6
                active = 'relu'
                active3 = 'softmax'
                inp1 = Input(shape=(None, None, N))
                bn = BatchNormalization()(inp1)
                l2 = Conv2D(64, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(bn)
                l3 = Conv2D(48, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(l2)
                l4 = Conv2D(32, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(l3)

                out = Conv2D(3, kernel_size=3, activation=active3, padding='same', kernel_initializer='he_normal')(l4)

                
            #    out= Conv2D(1, kernel_size=k_3, activation='relu', padding='same', kernel_initializer='he_normal',name="nothing")(out1)
                model = Model(inp1, out, name='shallow')
#                model = Model(inp, out, name='shallow')
            elif k_mod == "FPN":
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

            elif k_mod == "FractalNet":           
                f = 8
                # N = x_train.shape[-1]
                inputs = Input((size, size, N))
                active = 'relu'
                # conv1 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(inputs)
                conv1 = Conv2D(f, kernel_size=3, activation=active, padding='same', kernel_initializer = keras.initializers.glorot_normal(seed=None) )(inputs)
                conv1 = BatchNormalization()(conv1)
                # conv1 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv1)
                conv1 = Conv2D(f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv1)

                down1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            
                conv2 = BatchNormalization()(down1)
                # conv2 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv2)
                conv2 = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv2)
                conv2 = BatchNormalization()(conv2)
                # conv2 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv2)
                conv2 = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv2)
                
                down2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            
                conv3 = BatchNormalization()(down2)
                # conv3 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv3)
                conv3 = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv3)
                conv3 = BatchNormalization()(conv3)
                # conv3 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv3)
                conv3 = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv3)

                down3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            
                conv4 = BatchNormalization()(down3)
                # conv4 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv4)
                conv4 = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv4)
                conv4 = BatchNormalization()(conv4)
                # conv4 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv4)
                conv4 = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv4)


                down4 = MaxPooling2D(pool_size=(2, 2))(conv4)
            
                conv5 = BatchNormalization()(down4)
                # conv5 = Convolution2D(16 * f, 3, 3, activation='relu', border_mode='same')(conv5)
                conv5 = Conv2D(16*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv5)
                conv5 = BatchNormalization()(conv5)
                # conv5 = Convolution2D(16 * f, 3, 3, activation='relu', border_mode='same')(conv5)
                conv5 = Conv2D(16*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv5)

                # up1 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
                up1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
                conv6 = BatchNormalization()(up1)
                conv6 = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv6)
                # conv6 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6)
                conv6 = BatchNormalization()(conv6)
                # conv6 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6)
                conv6 = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv6)


                # up2 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
                up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
                conv7 = BatchNormalization()(up2)
                # conv7 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7)
                conv7 = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv7)
                conv7 = BatchNormalization()(conv7)
                # conv7 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7)
                conv7 = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv7)

                # up3 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
                up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)

                conv8 = BatchNormalization()(up3)
                # conv8 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv8)
                conv8 = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv8)
                conv8 = BatchNormalization()(conv8)
                # conv8 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv8)
                conv8 = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv8)

                # up4 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
                up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)

                conv9 = BatchNormalization()(up4)
                # conv9 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv9)
                conv9 = Conv2D(f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv9)
                conv9 = BatchNormalization()(conv9)
                # conv9 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv9)
                conv9 = Conv2D(f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv9)

                # --- end first u block
            
                down1b = MaxPooling2D(pool_size=(2, 2))(conv9)
                # down1b = merge([down1b, conv8], mode='concat', concat_axis=3)
                up2 = concatenate([down1b, conv8], axis=-1)

                conv2b = BatchNormalization()(down1b)
                # conv2b = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv2b)
                conv2b = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv2b)
                conv2b = BatchNormalization()(conv2b)
                # conv2b = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv2b)
                conv2b = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv2b)


                down2b = MaxPooling2D(pool_size=(2, 2))(conv2b)
                # down2b = merge([down2b, conv7], mode='concat', concat_axis=3)
                down2b = concatenate([down2b, conv7], axis=-1)
                
                conv3b = BatchNormalization()(down2b)
                # conv3b = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv3b)
                conv3b = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv3b)
                conv3b = BatchNormalization()(conv3b)
                # conv3b = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv3b)
                conv3b = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv3b)

                down3b = MaxPooling2D(pool_size=(2, 2))(conv3b)
                # down3b = merge([down3b, conv6], mode='concat', concat_axis=3)
                down3b = concatenate([down3b, conv6], axis=-1)

                conv4b = BatchNormalization()(down3b)
                # conv4b = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv4b)
                conv4b = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv4b)
                conv4b = BatchNormalization()(conv4b)
                # conv4b = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv4b)
                conv4b = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv4b)

                down4b = MaxPooling2D(pool_size=(2, 2))(conv4b)
                # down4b = merge([down4b, conv5], mode='concat', concat_axis=3)
                down4b = concatenate([down4b, conv5], axis=-1)

                conv5b = BatchNormalization()(down4b)
                # conv5b = Convolution2D(16 * f, 3, 3, activation='relu', border_mode='same')(conv5b)
                conv5b = Conv2D(16*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv5b)
                conv5b = BatchNormalization()(conv5b)
                # conv5b = Convolution2D(16 * f, 3, 3, activation='relu', border_mode='same')(conv5b)
                conv5b = Conv2D(16*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv5b)

                # up1b = merge([UpSampling2D(size=(2, 2))(conv5b), conv4b], mode='concat', concat_axis=3)
                up1b = concatenate([UpSampling2D(size=(2, 2))(conv5b), conv4b], axis=-1)


                conv6b = BatchNormalization()(up1b)
                # conv6b = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6b)
                conv6b = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv6b)
                conv6b = BatchNormalization()(conv6b)
                # conv6b = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6b)
                conv6b = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv6b)

                # up2b = merge([UpSampling2D(size=(2, 2))(conv6b), conv3b], mode='concat', concat_axis=3)
                up2b = concatenate([UpSampling2D(size=(2, 2))(conv6b), conv3b], axis=-1)
            
                conv7b = BatchNormalization()(up2b)
                # conv7b = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7b)
                conv7b = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv7b)
                conv7b = BatchNormalization()(conv7b)
                # conv7b = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7b)
                conv7b = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv7b)

                # up3b = merge([UpSampling2D(size=(2, 2))(conv7b), conv2b], mode='concat', concat_axis=3)
                up3b = concatenate([UpSampling2D(size=(2, 2))(conv7b), conv2b], axis=-1)

                conv8b = BatchNormalization()(up3b)
                # conv8b = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv8b)
                conv8b = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv8b)
                conv8b = BatchNormalization()(conv8b)
                # conv8b = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv8b)
                conv8b = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv8b)
                # up4b = merge([UpSampling2D(size=(2, 2))(conv8b), conv9], mode='concat', concat_axis=3)
                up4b = concatenate([UpSampling2D(size=(2, 2))(conv8b), conv9], axis=-1)


                conv9b = BatchNormalization()(up4b)
                # conv9b = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv9b)
                conv9b = Conv2D(f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv9b)
                conv9b = BatchNormalization()(conv9b)
                # conv9b = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv9b)
                conv9b = Conv2D(f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv9b)
                conv9b = BatchNormalization()(conv9b)
            
                # outputs = Convolution2D(3, 1, 1, activation=activate3, border_mode='same')(conv9b)
                outputs = Conv2D(3, kernel_size=1, activation='softmax', padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv9b)
                model = Model(inputs=inputs, outputs=outputs)                

            elif k_mod == "DenseUnet":           
                f = 8
                # N = x_train.shape[-1]
                inputs = Input((size, size, N))
                active = 'relu'
                # conv1 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(inputs)
                conv1 = Conv2D(f, kernel_size=3, activation=active, padding='same', kernel_initializer = keras.initializers.glorot_normal(seed=None) )(inputs)
                conv1 = BatchNormalization()(conv1)
                # conv1 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv1)
                conv1 = Conv2D(f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv1)

                down1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            
                conv2 = BatchNormalization()(down1)
                # conv2 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv2)
                conv2 = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv2)
                conv2 = BatchNormalization()(conv2)
                # conv2 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv2)
                conv2 = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv2)
                
                down2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            
                conv3 = BatchNormalization()(down2)
                # conv3 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv3)
                conv3 = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv3)
                conv3 = BatchNormalization()(conv3)
                # conv3 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv3)
                conv3 = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv3)

                down3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            
                conv4 = BatchNormalization()(down3)
                # conv4 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv4)
                conv4 = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv4)
                conv4 = BatchNormalization()(conv4)
                # conv4 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv4)
                conv4 = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv4)


                down4 = MaxPooling2D(pool_size=(2, 2))(conv4)
            
                conv5 = BatchNormalization()(down4)
                # conv5 = Convolution2D(16 * f, 3, 3, activation='relu', border_mode='same')(conv5)
                conv5 = Conv2D(16*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv5)
                conv5 = BatchNormalization()(conv5)
                # conv5 = Convolution2D(16 * f, 3, 3, activation='relu', border_mode='same')(conv5)
                conv5 = Conv2D(16*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv5)

                # up1 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
                up1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
                conv6 = BatchNormalization()(up1)
                conv6 = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv6)
                # conv6 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6)
                conv6 = BatchNormalization()(conv6)
                # conv6 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6)
                conv6 = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv6)


                # up2 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)

                up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
                up2 = concatenate([up2, UpSampling2D(size=(2, 2))(conv4)], axis=-1)
                conv7 = BatchNormalization()(up2)
                # conv7 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7)
                conv7 = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv7)
                conv7 = BatchNormalization()(conv7)
                # conv7 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7)
                conv7 = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv7)

                # up3 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
                up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
                up3 = concatenate([UpSampling2D(size=(2, 2))(conv3), up3], axis=-1)
                up3 = concatenate([UpSampling2D(size=(4,4))(conv4), up3], axis=-1)
                conv8 = BatchNormalization()(up3)
                # conv8 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv8)
                conv8 = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv8)
                conv8 = BatchNormalization()(conv8)
                # conv8 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv8)
                conv8 = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv8)

                # up4 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
                up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
                up4 = concatenate([UpSampling2D(size=(2, 2))(conv2), up4], axis=-1)
                up4 = concatenate([UpSampling2D(size=(4,4))(conv3), up4], axis=-1)
                up4 = concatenate([UpSampling2D(size=(8,8))(conv4), up4], axis=-1)
                conv9 = BatchNormalization()(up4)
                # conv9 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv9)
                conv9 = Conv2D(f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv9)
                conv9 = BatchNormalization()(conv9)
                # conv9 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv9)
                conv9 = Conv2D(f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv9)
            
                # outputs = Convolution2D(3, 1, 1, activation=activate3, border_mode='same')(conv9b)
                outputs = Conv2D(3, kernel_size=1, activation='softmax', padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv9)
                model = Model(inputs=inputs, outputs=outputs)                



            # model.load_weights(combinations + "_" + BACKBONE + "_" + name_model_pre + "_model_wIoU"+ str(size) +".h5")
            Adamax = Adam(lr=l_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            model.compile(loss=bce_jaccard_loss, metrics=[iou_score], optimizer=Adamax)
#            model.compile(loss='categorical_crossentropy', metrics=[iou_score], optimizer=Adamax)

######## TRAINING ############
#            model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
#            
            callbacks = [
                EarlyStopping(patience=10, verbose=1),
                ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
                ModelCheckpoint(combinations + "_" + BACKBONE + "_" + name_model + "_temporary_model_IoU"+ str(size) +".h5", verbose=1, save_best_only=True, save_weights_only=True),
                time_callback, tensorboard
            ]

#            dataAugmentaion = image.ImageDataGenerator(rotation_range = 30, zoom_range = 0.20, 
#            fill_mode = "nearest", shear_range = 0.20, horizontal_flip = True, 
#            width_shift_range = 0.1, height_shift_range = 0.1)
            train_val_p2 = n_batch*200 
            s_p_e = train_val_p2//n_batch
            val_p2 = n_batch*10
            val_pe = val_p2//n_batch
            print(type(s_p_e))
            model.fit_generator(train_generator(train_val_p2, n_batch,data_folder,folder_out_2, size,N), validation_data = val_generator(train_val_p2, n_batch,data_folder,folder_out_2, size,N), validation_steps = val_pe, steps_per_epoch = s_p_e, epochs = n_epochs,  callbacks = callbacks)#, class_weight=[0.8,0.1, 0.1])


###### fit 
#             model.fit(
# #                x=[np.reshape(x_train[:,:,:,:5],newshape=(y_train.shape[0],y_train.shape[1],y_train.shape[2],5)),np.reshape(x_train[:,:,:,5],newshape=(y_train.shape[0],y_train.shape[1],y_train.shape[2],1))], #reverse dimensions (in load_data_SR) and input from y_train
#                 x=x_train,
#                 y=y_train,
#                 batch_size=n_batch,
#                 epochs=n_epochs,
# #                class_weight = class_weight,
# #                validation_data=([np.reshape(x_val[:,:,:,:5],newshape=(x_val.shape[0],x_val.shape[1],x_val.shape[2],5)), np.reshape(x_val[:,:,:,5],newshape=(x_val.shape[0],x_val.shape[1],x_val.shape[2],1))], y_val),callbacks = callbacks
#                 validation_data=(x_val, y_val),callbacks = callbacks
#             )


            ## fit model
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
            model.save_weights(combinations + "_" + BACKBONE + "_" + name_model + "_model_wIoU"+ str(size) +".h5")
  ############END TRAINING#############           
            
            # Load best model
            model.load_weights(combinations + "_" + BACKBONE + "_" + name_model + "_model_wIoU"+ str(size) +".h5")

            x_test = np.ndarray(shape=(10,size,size,6))
            y_test = np.ndarray(shape=(10,size,size,3))
            preds_val = np.ndarray(shape=(10,size,size,3), dtype='float32')
            k = 0
            for c in range(0,10): #test_p, test_p//10):
                H = np.load(os.path.join(data_folder, 'X_test_' + str(c) + '.npy'))                       
                x_test[k,:,:,:] = np.reshape(H[:,:, :], newshape=(size,size,6))
                y_test[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_test_' + str(c) + '.npy'))
                k += 1

#                preds_train = model.predict(x_train, verbose=1)
            preds_val1 = model.predict(x_test, verbose=1)
            mix_LAYERS_2 = np.squeeze(preds_val1)
            mix_LAYERS_2 = np.argmax(mix_LAYERS_2, axis = -1)
            
            preds_val[:,:,:,0] = mix_LAYERS_2 == 0
            preds_val[:,:,:,1] = mix_LAYERS_2 == 1
            preds_val[:,:,:,2] = mix_LAYERS_2 == 2

            
#            preds_val1 = model.predict([np.reshape(x_test[:,:,:,:5],newshape=(x_test.shape[0],x_test.shape[1],x_test.shape[2],5)), np.reshape(x_test[:,:,:,5],newshape=(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))], verbose=1)
#            pp = np.squeeze(preds_val1)
##            if n_shapes == 128:
#            preds_val = np.argmax(pp, axis = -1)
#            preds_val = model.predict_generator(x_val)        
#            preds_val = np.argmax(preds_val, axis=-1) #multiple categories
            # preds_val = preds_val1
            for k in range(0,10):#,int(x_test.shape[0]/10)): 
                x_val_1 = x_test[k,:,:,:]
                y_val_1 = y_test[k,:,:,:]
                pred_val_1 = preds_val[k,:,:,:]
                pred_val_2 = preds_val1[k,:,:,:]

                ndvi = y_val_1.astype('float32')
                im = combinations + "_" + BACKBONE + '_' + name_model + '_target_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                imsave(im, ndvi)
                
                ndvi2 = pred_val_1.astype('float32')
                im2 = combinations + "_" + BACKBONE + '_' + name_model + '_output_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                imsave(im2, ndvi2)

                ndvi2 = pred_val_2.astype('float32')
                im2 = combinations + "_" + BACKBONE + '_' + name_model + '_output_wo_max_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                imsave(im2, ndvi2)

                ndvi4 = x_val_1[:,:,5].astype('float32')
                im3 = combinations + "_" + BACKBONE + '_' + name_model + '_ndvi_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                imsave(im3, ndvi4)

                
#                if combinations == "case_5": 
#                    ndvi3 = x_val_1[:,:,0].astype('float32')
                    # im3 = combinations + "_" + BACKBONE + '_' + name_model + '_gamma0dB_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                    # imsave(im3, ndvi3)
                    # ndvi4 = x_val_1[:,:,1].astype('float32')
#                    im4 =  combinations + "_" + BACKBONE + '_' + name_model + '_rhoLT_'+ str(k) + '_wpatches'+ str(size) +'.tif'
#                    imsave(im4, ndvi4)
#                    ndvi3 = x_val_1[:,:,2].astype('float32')
#                    im3 = combinations + "_" + BACKBONE + '_' + name_model + '_tau_'+ str(k) + '_wpatches'+ str(size) +'.tif'
#                    imsave(im3, ndvi3)
#                    ndvi4 = x_val_1[:,:,3].astype('float32')
#                    im4 =  combinations + "_" + BACKBONE + '_' + name_model + '_localthetainc_'+ str(k) + '_wpatches'+ str(size) +'.tif'
#                    imsave(im4, ndvi4)
#                    ndvi4 = x_val_1[:,:,4].astype('float32')
#                    im4 =  combinations + "_" + BACKBONE + '_' + name_model + '_rho_6_'+ str(k) + '_wpatches'+ str(size) +'.tif'
#                    imsave(im4, ndvi4)
#                    ndvi4 = x_val_1[:,:,5].astype('float32')
#                    im4 =  combinations + "_" + BACKBONE + '_' + name_model + '_NDVI_'+ str(k) + '_wpatches'+ str(size) +'.tif'
#                    imsave(im4, ndvi4)

        #    model.evaluate(x_val, y_val, verbose=1)
            # Predict on train, val and test
#            if name_model == "PSPNet":
#                preds_train = model.predict(x_train2, verbose=1)
#                preds_val = model.predict(x_val2, verbose=1)
#                for k in range(0,x_val.shape[0],int(x_val.shape[0]/100)):
#                    x_val_1 = x_val2[k,:,:,:]
#                    y_val_1 = y_val2[k,:,:,:]
#                    pred_val_1 = preds_val[k,:,:,:]
#                    
#                    ndvi = y_val_1.astype('float32')
#                    im = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_target_'+ str(k) + '_wpatches'+ str(size) +'.tif'
#                    imsave(im, ndvi)
#                    
#                    ndvi2 = pred_val_1.astype('float32')
#                    im2 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_output_'+ str(k) + '_wpatches'+ str(size) +'.tif'
#                    imsave(im2, ndvi2)
#                    if N == 1:
#                        ndvi3 = x_val_1[:,:,0].astype('float32')
#                        im3 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
#                        imsave(im3, ndvi3)
#                    else: 
#                        ndvi3 = x_val_1[:,:,0].astype('float32')
#                        im3 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
#                        imsave(im3, ndvi3)
#                        ndvi4 = x_val_1[:,:,1].astype('float32')
#                        im4 =  comb[combinations] + "_" + BACKBONE + '_' + name_model + '_VH_'+ str(k) + '_wpatches'+ str(size) +'.tif'
#                        imsave(im4, ndvi4)
#            else:
##                preds_train = model.predict(x_train, verbose=1)
#                preds_val = model.predict(x_val, verbose=1)
#                
#                for k in range(0,x_val.shape[0],int(x_val.shape[0]/100)): 
#                    x_val_1 = x_val[k,:,:,:]
#                    y_val_1 = y_val[k,:,:,:]
#                    pred_val_1 = preds_val[k,:,:,:]
#                    
#                    ndvi = y_val_1.astype('float32')
#                    im = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_target_'+ str(k) + '_wpatches'+ str(size) +'.tif'
#                    imsave(im, ndvi)
#                    
#                    ndvi2 = pred_val_1.astype('float32')
#                    im2 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_output_'+ str(k) + '_wpatches'+ str(size) +'.tif'
#                    imsave(im2, ndvi2)
#                    if N == 1: 
#                        ndvi3 = x_val_1[:,:,0].astype('float32')
#                        im3 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
#                        imsave(im3, ndvi3)
#                    else: 
#                        ndvi3 = x_val_1[:,:,0].astype('float32')
#                        im3 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
#                        imsave(im3, ndvi3)
#                        ndvi4 = x_val_1[:,:,1].astype('float32')
#                        im4 =  comb[combinations] + "_" + BACKBONE + '_' + name_model + '_VH_'+ str(k) + '_wpatches'+ str(size) +'.tif'
#                        imsave(im4, ndvi4)
#
