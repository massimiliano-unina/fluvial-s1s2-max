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

from segmentation_models import Unet,PSPNet,Linknet,FPN#, shallow
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from segmentation_models.load_data import load_data,combinations_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,Callback
from keras.layers import Conv2D, Input, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
import time
import numpy as np
from tifffile import imsave
from mat4py import savemat, loadmat
import argparse

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

## testing a parte 
size = 32
size_t = 144 #, 128,128]# 128
n_epochs = 5
n_batch = 64
l_rate = 0.001

comb = ["water_Po"]#["VH","VV","VVaVH", "VVaVHaSum","VVaVHaRatio", "VVaVHaDiff","Total"]#"Ratio",  #["VVaVH"]#Ratio"]# ["VVaVH"]##
num = [3] #[1,1,2,3,3,3,5]#1,#[2]# 

#class_weight = {0: 50., 1: 1., 2: 1.}
#class_weight = [40.0, 1.0, 1.0]

#x_traina, y_traina, x_vala, y_vala,x_train2a, y_train2a, x_val2a, y_val2a = load_data(folder_1, size,size_t,indo,indo1)
#print(x_traina.shape)
#print(x_vala.shape)

#train_val = np.load("train_data.npz")
train_val = np.load("train_data_water_Po.npz")
n_classes = 1
x_train1 = train_val['x_train']
print(x_train1.shape)
y_train1 = train_val['y_train']
x_val1 = train_val['x_val']
y_val1 = train_val['y_val']
x_test1 = train_val['x_test']
y_test1 = train_val['y_test']

N1 = x_train1.shape[-1]

x_train = np.ndarray(shape=(n_batch*1125, size, size, N1 ), dtype='float32')
x_train = x_train1[:n_batch*1125, :size,:size,:]

del x_train1
y_train = y_train1[:n_batch*1125, :size,:size,:n_classes]
del y_train1
x_val = np.ndarray(shape=(x_val1.shape[0], size, size, N1 ), dtype='float32')
x_val[:,:,:, :] = x_val1[:, :size,:size,:]
#x_val[:,:,:, 3] = x_val1[:, :size,:size,4]
#x_val = x_val1[:, :size,:size,:]
del x_val1
y_val = y_val1[:, :size,:size,:n_classes]
del y_val1
N = x_train.shape[-1]
print(x_train.shape)
print(x_val.shape)
x_test = x_test1[:, :,:,:]
del x_test1
y_test = y_test1[:, :,:,:]
del y_test1


for combinations in range(len(comb)):
    
    backs = ['mobilenetv2']# ['resnet34','mobilenetv2']
    for k_back in range(len(backs)):
        BACKBONE = backs[k_back]# 
        preprocess_input = get_preprocessing(BACKBONE)        ## define model and chose between following models: 
        
        networks =   ["Shallow"] #["FPN"] #["Unet"]# ["Unet","Linknet","FPN"]# ,"PSPNet","Linknet","FPN"]#["PSPNet","Linknet","FPN"]# 
        for k_mod in range(len(networks)):
        #    size = size_t[k_mod]
            name_model = networks[k_mod]+"6"#+ "_despeck3_"# +"3"#
            if k_mod == 1:
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
                
            elif k_mod == 0:
                N = x_train.shape[-1]
                active = 'relu'
                active3 = 'sigmoid'
                inp = Input(shape=(None, None, 2))
                bn = BatchNormalization()(inp)
#                l1 = Conv2D(64, kernel_size=3, activation= active, padding='same', kernel_initializer='he_normal' )(bn)
#                bn1 = BatchNormalization()(l1)

                l2 = Conv2D(48, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(bn)
                l3 = Conv2D(32, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(l2)
                out = Conv2D(1, kernel_size=3, activation=active3, padding='same', kernel_initializer='he_normal')(l3)
                
            #    out= Conv2D(1, kernel_size=k_3, activation='relu', padding='same', kernel_initializer='he_normal',name="nothing")(out1)
                model = Model(inp, out, name='shallow')
                
            elif k_mod == 2:
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
                    
            Adamax = Adam(lr=l_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            model.compile(loss=bce_jaccard_loss, metrics=[iou_score], optimizer=Adamax)
#            model.compile(loss='categorical_crossentropy', metrics=[iou_score], optimizer=Adamax)

######## TRAINING ############
#            model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
#            
            callbacks = [
                EarlyStopping(patience=10, verbose=1),
                ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
                ModelCheckpoint(comb[combinations] + "_" + BACKBONE + "_" + name_model + "_temporary_model_IoU"+ str(size) +".h5", verbose=1, save_best_only=True, save_weights_only=True),
                time_callback
            ]

#            dataAugmentaion = image.ImageDataGenerator(rotation_range = 30, zoom_range = 0.20, 
#            fill_mode = "nearest", shear_range = 0.20, horizontal_flip = True, 
#            width_shift_range = 0.1, height_shift_range = 0.1)
#            model.fit_generator(dataAugmentaion.flow(x_train, y_train, batch_size = n_batch), validation_data = (x_val, y_val), steps_per_epoch = len(x_train) // n_batch, epochs = n_epochs)

            model.fit(
                x=x_train,
                y=y_train,
                batch_size=n_batch,
                epochs=n_epochs,
#                class_weight = class_weight,
                validation_data=(x_val, y_val),callbacks = callbacks
            )


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
            model.save_weights(comb[combinations] + "_" + BACKBONE + "_" + name_model + "_model_wIoU"+ str(size) +".h5")
 ############END TRAINING#############           
            
            # Load best model
            model.load_weights(comb[combinations] + "_" + BACKBONE + "_" + name_model + "_model_wIoU"+ str(size) +".h5")

#                preds_train = model.predict(x_train, verbose=1)
            preds_val1 = model.predict(x_test, verbose=1)
#            pp = np.squeeze(preds_val1)
##            if n_shapes == 128:
#            preds_val = np.argmax(pp, axis = -1)
#            preds_val = model.predict_generator(x_val)        
#            preds_val = np.argmax(preds_val, axis=-1) #multiple categories
            preds_val = preds_val1
            for k in range(0,x_test.shape[0]):#,int(x_test.shape[0]/10)): 
                x_val_1 = x_test[k,:,:,:]
                y_val_1 = y_test[k,:,:,:]
                y_val_1 = np.reshape(y_val_1, ( y_val_1.shape[0], y_val_1.shape[1]))

                print(y_val_1.shape)
                pred_val_1 = preds_val[k,:,:,:] > 0
                pred_val_1 = np.reshape(pred_val_1, (pred_val_1.shape[0], pred_val_1.shape[1]))
                print(pred_val_1.shape)
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
                    im3 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_gamma0dB_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                    imsave(im3, ndvi3)
                    ndvi4 = x_val_1[:,:,1].astype('float32')
                    im4 =  comb[combinations] + "_" + BACKBONE + '_' + name_model + '_rhoLT_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                    imsave(im4, ndvi4)
#                    ndvi4 = x_val_1[:,:,2].astype('float32')
#                    im4 =  comb[combinations] + "_" + BACKBONE + '_' + name_model + '_tau_'+ str(k) + '_wpatches'+ str(size) +'.tif'
#                    imsave(im4, ndvi4)
#                    ndvi4 = x_val_1[:,:,3].astype('float32')
#                    im4 =  comb[combinations] + "_" + BACKBONE + '_' + name_model + '_localthetainc_'+ str(k) + '_wpatches'+ str(size) +'.tif'
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
