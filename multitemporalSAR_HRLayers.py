# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from segmentation_models import Unet,PSPNet,Linknet,FPN
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from segmentation_models.load_data import load_data,combinations_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,Callback
from keras.layers import Conv2D, Input, BatchNormalization, UpSampling2D, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.callbacks import TensorBoard

from util import train_generator_2, val_generator_2
# from keras.callbacks import TensorBoard
import time
import numpy as np
from tifffile import imsave
from mat4py import savemat, loadmat
import argparse
import os
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
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)

# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
# indo = loadmat(r"C:\Users\massi\Downloads\segmentation_models-master\docs\indices.mat")
# indo1 = loadmat(r"C:\Users\massi\Downloads\segmentation_models-master\docs\indices1.mat")
# #
# #indices = 0#indo['A']
time_callback = TimeHistory()

    
# load your data
## training and validation normale
folder_1 = r"D:\Works\Albufera-SemanticSegmentation\S2_S1\\" #r"D:\Albufera\\"#
folder_train = r"D:\Albufera_2019_processed\Training\Sigma\\"

dir_list = os.listdir(folder_train)
dir_list.sort()

number_train = 0
number_val = 0
size = 128
size_t = 144 #, 128,128]# 128
n_epochs = 5# 10 #10
n_batch = 32
N = 6



# x_train = np.ndarray(shape=(0, size, size, N), dtype='float32')
# y_train = np.ndarray(shape=(0, size, size, 3), dtype='float32')
# x_val = np.ndarray(shape=(0, size, size, N), dtype='float32')
# y_val = np.ndarray(shape=(0, size, size, 3), dtype='float32')

# timer = time.time()
# for file_1 in dir_list: 
#     if file_1.find("X_train_") != -1:
#         X = np.load(os.path.join(folder_train, 'X_train_' + str(number_train) + '.npy'))
#         x_1 = np.reshape(X[:,:,:], newshape=(1,size,size,N))
#         Y = np.load(os.path.join(folder_train, 'Y_train_' + str(number_train) + '.npy'))
#         y_1 = np.reshape(Y[:,:,:3], newshape=(1,size,size,3))
#         x_train = np.concatenate((x_train, x_1))
#         y_train = np.concatenate((y_train, y_1))
#         print(x_train.shape)
#         number_train += 1
#     if file_1.find("X_val_") != -1:
#         X_ = np.load(os.path.join(folder_train, 'X_val_' + str(number_val) + '.npy'))
#         x_1_ = np.reshape(X_[:,:,:], newshape=(1,size,size,N))
#         Y_ = np.load(os.path.join(folder_train, 'Y_val_' + str(number_val) + '.npy'))
#         y_1_ = np.reshape(Y_[:,:,:3], newshape=(1,size,size,3))
#         x_val = np.concatenate((x_val, x_1_))
#         y_val = np.concatenate((y_val, y_1_))
#         print(x_val.shape)
#         number_val += 1
#         timer2 = time.time() - timer
# print(number_train)
# print(number_val)

# from joblib import Parallel, delayed
# def compute_dir_list(y, folder_train,k):#(x_val, x_train, y_val, y_train, folder_train, k,number_train, number_val):
#     # for file_1 in dir_list[k:k+1]: 
#     x_val = y[0]
#     x_train = y[1]
#     y_val = y[2]
#     y_train = y[3]
#     number_train = y[4]
#     number_val = y[5]
#     if k.find("X_train_") != -1:
#         X = np.load(os.path.join(folder_train, 'X_train_' + str(number_train) + '.npy'))
#         x_1 = np.reshape(X[:,:,:], newshape=(1,size,size,N))
#         Y = np.load(os.path.join(folder_train, 'Y_train_' + str(number_train) + '.npy'))
#         y_1 = np.reshape(Y[:,:,:3], newshape=(1,size,size,3))
#         x_train = np.concatenate((x_train, x_1))
#         y_train = np.concatenate((y_train, y_1))
#         print(x_train.shape)
#         number_train += 1
#     if k.find("X_val_") != -1:
#         X_ = np.load(os.path.join(folder_train, 'X_val_' + str(number_val) + '.npy'))
#         x_1_ = np.reshape(X_[:,:,:], newshape=(1,size,size,N))
#         Y_ = np.load(os.path.join(folder_train, 'Y_val_' + str(number_val) + '.npy'))
#         y_1_ = np.reshape(Y_[:,:,:3], newshape=(1,size,size,3))
#         x_val = np.concatenate((x_val, x_1_))
#         y_val = np.concatenate((y_val, y_1_))
#         print(x_val.shape)
#         number_val += 1
#     return (x_val, x_train, y_val, y_train, number_train, number_val)
#     print(number_train)
#     print(number_val)

# x_train = np.ndarray(shape=(0, size, size, N), dtype='float32')
# y_train = np.ndarray(shape=(0, size, size, 3), dtype='float32')
# x_val = np.ndarray(shape=(0, size, size, N), dtype='float32')
# y_val = np.ndarray(shape=(0, size, size, 3), dtype='float32')
# y = (x_val, x_train, y_val, y_train, number_train, number_val)
# y = Parallel(n_jobs=6)(delayed(compute_dir_list)(y,folder_train,i) for i in dir_list)
# x_val = y[0]
# x_train = y[1]
# y_val = y[2]
# y_train = y[3]
# number_train = y[4]
# number_val = y[5]


## testing a parte 
#folder_1 = r"D:\Works\Albufera-SemanticSegmentation\Testing\S2_S1\\"
comb = ["TriGamma"] #["TriVH","TriVV","TriVVaVH"]# ["VH","VV","VVaVH"]#, "VVaVHaSum","VVaVHaRatio", "VVaVHaDiff","Total"]#"Ratio",  #["VVaVH"]#Ratio"]# ["VVaVH"]##
num = [6]# [3,3,6] # [1,1,2]#,3,3,3,5]#1,#[2]# 

#class_weight = {0: 50., 1: 1., 2: 1.}
class_weight = [40.0, 1.0, 1.0]

#x_traina, y_traina, x_vala, y_vala,x_train2a, y_train2a, x_val2a, y_val2a = load_data(folder_1, size,size_t,indo,indo1)
train_val =  np.load("test_data_SAR_Iodice_gamma_with_summer_augmentation.npz")#,x_test = x_test, y_test= y_test, x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val)    
# train_val =  np.load("test_data_SAR_Iodice.npz")#,x_test = x_test, y_test= y_test, x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val)    
# x_train = train_val['x_train']
# y_train = train_val['y_train']
# x_val = train_val['x_val']
# y_val = train_val['y_val']
x_test_2 = train_val['x_gtest']
y_test_2 = train_val['y_test']
#
#comb = ["TriVV","TriVVaVH"]# ["VH","VV","VVaVH"]#, "VVaVHaSum","VVaVHaRatio", "VVaVHaDiff","Total"]#"Ratio",  #["VVaVH"]#Ratio"]# ["VVaVH"]##
#num = [3,6]# [1,1,2]#,3,3,3,5]#1,#[2]# 


for combinations in comb:
    # if combinations == "TriVV": 
    #     x_train,  x_val, x_test = x_train_2[:,:,:,:3],  x_val_2[:,:,:,:3], x_test_2[:,:,:,:3]
    #     y_train,  y_val, y_test = y_train_2, y_val_2, y_test_2
    # elif combinations == "TriVH": 
    #     x_train, x_val, x_test =  x_train_2[:,:,:,3:],  x_val_2[:,:,:,3:], x_test_2[:,:,:,3:]
    #     y_train,  y_val, y_test = y_train_2, y_val_2, y_test_2
    # elif combinations == "TriGamma": #"TriVVaVH": 
    #     x_train,  x_val, x_test =   x_train_2[:,:,:,:],  x_val_2[:,:,:,:], x_test_2[:,:,:,:]
    #     y_train,  y_val, y_test = y_train_2, y_val_2, y_test_2
    # N = x_train.shape[-1]
    # print(x_train.shape)
    # print(x_val.shape)
    backs = ['mobilenetv2']# ['resnet34','mobilenetv2']
    for k_back in range(len(backs)):
        BACKBONE = backs[k_back]# 
        preprocess_input = get_preprocessing(BACKBONE)        ## define model and chose between following models: 
        
        networks = ["shallow"] # ["Unet","Linknet","FPN", "shallow"]# ["Unet"]#,"PSPNet","Linknet","FPN"]#["PSPNet","Linknet","FPN"]# 
        for k_mod in networks: # range(len(networks)):
        #    size = size_t[k_mod]
            name_model = k_mod + "6" # networks[k_mod]+"5"#+ "_despeck3_"# +"3"#
            name_model = k_mod +"_HRL"#+ "_despeck3_"# +"3"#
            name_model = name_model + "_" + combinations + "_{}".format(int(time.time()))
            tensorboard = TensorBoard(log_dir='logs/{}'.format(name_model))

            if k_mod == "Unet":
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
                
            elif k_mod == "Linknet":
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
                
            elif k_mod == "shallow":
                # N = x_train.shape[-1]
                active = 'relu'
                active3 = 'softmax'
                inp = Input(shape=(None, None, N))
                bn = BatchNormalization()(inp)
#                l1 = Conv2D(64, kernel_size=3, activation= active, padding='same', kernel_initializer='he_normal' )(bn)
#                bn1 = BatchNormalization()(l1)

                l2 = Conv2D(64, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(bn)
                l3 = Conv2D(48, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(l2)
                l4 = Conv2D(32, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(l3)
                # out2 = Conv2D(2, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal', name= "L1")(l4)
                # l5 = Conv2D(32, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(out2)
                out = Conv2D(3, kernel_size=3, activation=active3, padding='same', kernel_initializer='he_normal', name="Class")(l4) #(l5)
                
            #    out= Conv2D(1, kernel_size=k_3, activation='relu', padding='same', kernel_initializer='he_normal',name="nothing")(out1)
                model = Model(inp, out, name='shallow') # [out2, out], name='shallow')
                # model = multi_gpu_model(model, 2)
            elif k_mod == "deeper":
                # N = x_train.shape[-1]
                active = 'relu'
                active3 = 'softmax'
                inp = Input(shape=(None, None, N))
                bn = BatchNormalization()(inp)
#                l1 = Conv2D(64, kernel_size=3, activation= active, padding='same', kernel_initializer='he_normal' )(bn)
#                bn1 = BatchNormalization()(l1)

                l2 = Conv2D(64, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(bn)
                l3 = Conv2D(48, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(l2)
                l4 = concatenate([l3, inp], axis=-1)
#                bn2 = BatchNormalization()(l4)
                l5 = Conv2D(32, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(l4)

                out = Conv2D(3, kernel_size=3, activation=active3, padding='same', kernel_initializer='he_normal')(l5)
                
            #    out= Conv2D(1, kernel_size=k_3, activation='relu', padding='same', kernel_initializer='he_normal',name="nothing")(out1)
                model = Model(inp, out, name='shallow')


            elif k_mod == "FPN":
                # N = x_train.shape[-1]
                if N == 3: 
                    model = FPN(BACKBONE, input_shape=(size, size, 3), classes=3, activation='softmax', encoder_weights='imagenet', encoder_freeze=False)
                else:
                    base_model = FPN(BACKBONE, input_shape=(size, size, 3), classes=3, activation='softmax', encoder_weights='imagenet', encoder_freeze=False)
                    inp = Input(shape=(size, size, N))
                    bn = BatchNormalization()(inp)
                    l1 = Conv2D(3, (1, 1))(bn) # map N channels data to 3 channels
                    out = base_model(l1)
                    model = Model(inp, out, name=base_model.name)

######### TRAINING ############
            Adamax = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            model.compile(loss=bce_jaccard_loss, metrics=[iou_score], optimizer=Adamax)
#            model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
#            
            callbacks = [
                EarlyStopping(patience=10, verbose=1),
                ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
                ModelCheckpoint(combinations + "_" + BACKBONE + "_" + name_model + "_temporary_model_IoU"+ str(size) +".h5", verbose=1, save_best_only=True, save_weights_only=True),
                time_callback
            ]
            
            ## fit model
            # if name_model == "PSPNet":
            #     model.fit(
            #         x=x_train,
            #         y=y_train,
            #         batch_size=n_batch,
            #         epochs=n_epochs,
            #         class_weight = class_weight,
            #         validation_data=(x_val, y_val),callbacks = callbacks
            #     )
            # else: 
            #     model.fit(
            #         x=x_train,
            #         y=y_train,
            #         batch_size=n_batch,
            #         epochs=n_epochs,
            #         class_weight = class_weight,
            #         validation_data=(x_val, y_val),callbacks = callbacks
            #     )
            train_val_p2 = n_batch*50 
            s_p_e = train_val_p2//n_batch
            val_p2 = n_batch*10
            val_pe = val_p2//n_batch
            print(type(s_p_e))
            model.fit_generator(train_generator_2(train_val_p2, number_train, n_batch,folder_train, size,N), validation_data = val_generator_2(train_val_p2, number_val, n_batch,folder_train, size,N), validation_steps = val_pe, steps_per_epoch = s_p_e, epochs = n_epochs)
            
            times = time_callback.times
            dic_times = {}
            dic_times['times'] = times
            savemat(combinations + "_" + BACKBONE + '_' + name_model + '_times.mat', dic_times)
            model.save_weights(combinations + "_" + BACKBONE + "_" + name_model + "_model_wIoU"+ str(size) +".h5")
 ############END TRAINING#############           
            
            # Load best model
            model.load_weights(combinations+ "_" + BACKBONE + "_" + name_model + "_model_wIoU"+ str(size) +".h5")
            
        #    model.evaluate(x_val, y_val, verbose=1)
            # Predict on train, val and test
            if name_model == "PSPNet":
                preds_train = model.predict(x_train2, verbose=1)
                preds_val = model.predict(x_val2, verbose=1)
                for k in range(0,x_val.shape[0],int(x_val.shape[0]/100)):
                    x_val_1 = x_val2[k,:,:,:]
                    y_val_1 = y_val2[k,:,:,:]
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
            else:
#                preds_train = model.predict(x_train, verbose=1)
                y_test_2 = y_test_2[:,:,:,:3]
                preds_val = np.ndarray(shape = y_test_2.shape, dtype='float32')
                for a in range(0,x_test_2.shape[1],500):
                    for b in range(0,x_test_2.shape[2],500):
                        preds_val[:,a:a+500, b:b+500, :] = model.predict(x_test_2[:,a:a+500, b:b+500, :], verbose=1)
                mix_LAYERS_2 = np.squeeze(preds_val)
                mix_LAYERS_2 = np.argmax(mix_LAYERS_2, axis = -1)
                
                preds_val[:,:,:,0] = mix_LAYERS_2 == 0
                preds_val[:,:,:,1] = mix_LAYERS_2 == 1
                preds_val[:,:,:,2] = mix_LAYERS_2 == 2
                
                
                for k in range(0,x_test_2.shape[0]): 
                    x_val_1 = x_test_2[k,:,:,:]
                    y_val_1 = y_test_2[k,:,:,:]
                    pred_val_1 = preds_val[k,:,:,:]
                    
                    ndvi = y_val_1.astype('float32')
                    im = combinations + "_" + BACKBONE + '_' + name_model + '_target_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                    imsave(im, ndvi)
                    
                    ndvi2 = pred_val_1.astype('float32')
                    im2 = combinations + "_" + BACKBONE + '_' + name_model + '_output_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                    imsave(im2, ndvi2)
                    if N == 1: 
                        ndvi3 = x_val_1[:,:,0].astype('float32')
                        im3 = combinations + "_" + BACKBONE + '_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                        imsave(im3, ndvi3)
                    else: 
                        ndvi3 = x_val_1[:,:,0].astype('float32')
                        im3 = combinations + "_" + BACKBONE + '_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                        imsave(im3, ndvi3)
                        ndvi4 = x_val_1[:,:,1].astype('float32')
                        im4 =  combinations + "_" + BACKBONE + '_' + name_model + '_VH_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                        imsave(im4, ndvi4)
