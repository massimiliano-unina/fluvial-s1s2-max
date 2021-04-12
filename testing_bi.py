# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from segmentation_models import shallow_CNN2#,Unet,PSPNet,Linknet,FPN
#from segmentation_models.backbones import get_preprocessing
#from segmentation_models.losses import bce_jaccard_loss
from keras.losses import mean_absolute_error
from keras.optimizers import Adam
#from segmentation_models.metrics import iou_score
from segmentation_models.load_data import load_dataset,load_data_super_resolution,combinations_input_super_resolution
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,Callback
from keras.layers import Conv2D, Input
from keras.models import Model
import time
import numpy as np
from tifffile import imsave
from mat4py import savemat, loadmat
import pickle
import os 
#from scipy.io import sio
#import argparse

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

from others import parser_xml
#xml_file=parser_xml(r"C:\Users\massi\Downloads\segmentation_models-master\config.xml")


indo = loadmat(r"D:\Dataset_SuperRisoluzione\indices.mat")
#indices = _['A']
indices = 0
time_callback = TimeHistory()

    
# load your data
folder_1 = r"D:\Dataset_MaxAntonio\\" #r"D:\New folder1\\"
MODELS_PATH = r"C:\Users\massi\Downloads\segmentation_models-master\docs\model_path\\"

ts = str(int(time.time())) #Time stamp for the model distinctness
model_path = os.path.join(MODELS_PATH, 'model-{}.h5'.format(ts))
history_path = os.path.join(MODELS_PATH, 'model-{}.history'.format(ts))

size = 128# 33
n_epochs = 200
n_batch = 1

num_bands = 10
k_1 = 9
k_2 = 5
k_3 = 5

sizer = [128,256,512] 
dims = ["small","medium","large"]
zone = ["Sidney","Tokyo","Adis_Abeba","Athens"]#"New York",
band = ['05','06','07','8A','11','12']
ps_first = 512
patches = [3,3,3,3]#,2
for city in range(len(zone)):
    city_name = zone[city]
    patchette = patches[city]
    for patch0 in range(patchette):
        for dimensions in range(len(dims)): 
            preambol_name = dims[dimensions]
            ps = sizer[dimensions]
            k_p_p = 0
            Ts = ps # 2*ps
            Ts_2 = ps//2
            k = 0
            for combinations in band:
                print(city_name + '_' + preambol_name + '_' + str(patch0) + '_B' +combinations)
                lr_starting = 0.0001
                X_traina, y_traina, X_vala, y_vala, X_testa, y_testa, X_test2a, y_test2a = load_dataset(folder_1, ps_first, 0, combinations, num_bands, city_name, patch0)                 
                x_train, y_train, x_val, y_val, x_test, y_test, x_test2, y_test2 = X_traina[:,:Ts_2,:Ts_2,:], y_traina[:,:Ts_2,:Ts_2,:], X_vala[:,:Ts_2,:Ts_2,:], y_vala[:,:Ts_2,:Ts_2,:], X_testa[:,:Ts_2,:Ts_2,: ], y_testa[:,:Ts_2,:Ts_2 ,:], X_test2a[:,:Ts,:Ts ,: ], y_test2a[:,:Ts ,:Ts ,:]                
#                    x_train, y_train, x_val, y_val = load_data_super_resolution(folder_1, size,combinations,num_bands,indices)
#            #    x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = combinations_input_super_resolution(x_traina, y_traina, x_vala, y_vala,x_train2a, y_train2a, x_val2a, y_val2a, comb[combinations],num[combinations] )
                N = x_train.shape[-1]
#                print(x_train.shape)
#                print(x_val.shape)
            
                model = shallow_CNN2(num_bands, k_1, k_2, k_3)
                model.load_weights(MODELS_PATH + "model_weigths"+ combinations+ ".h5")
#                
#                Adamax = Adam(lr=lr_starting, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#                model.compile( loss=mean_absolute_error, metrics=[mean_absolute_error], optimizer=Adamax)
#                #'Adam',
#                
#                ## fit model
#                history2 = LossHistory()
#                
#                callbacks = [
#                    EarlyStopping(patience=10, verbose=1),
#                    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
#                    ModelCheckpoint("temporary_model_"+ combinations+ ".h5", verbose=1, save_best_only=True, save_weights_only=True),
#                    time_callback,
#                    history2
#                ]
#                
#                history = model.fit(
#                    x=[x_train,np.reshape(y_train[:,:,:,1],newshape=(y_train.shape[0],y_train.shape[1],y_train.shape[2],1))], #reverse dimensions (in load_data_SR) and input from y_train
#                    y=np.reshape(y_train[:,:,:,0],newshape=(y_train.shape[0],y_train.shape[1],y_train.shape[2],1)),
#                    batch_size=n_batch,
#                    epochs=n_epochs,
#                    validation_data=([x_val,np.reshape(y_val[:,:,:,1],newshape=(y_val.shape[0],y_val.shape[1],y_val.shape[2],1))],np.reshape(y_val[:,:,:,0],newshape=(y_val.shape[0],y_val.shape[1],y_val.shape[2],1))),callbacks = callbacks
#                )
#                model.save(MODELS_PATH + city_name + '_' + preambol_name + '_' + str(patch0) +"model_"+ combinations+ ".h5")
#                model.save_weights(MODELS_PATH + city_name + '_' + preambol_name + '_' + str(patch0) + "model_weigths"+ combinations+ ".h5")

                # Load best model
#                model.load_weights(MODELS_PATH + city_name + '_' + preambol_name + '_' + str(patch0) + "model_weigths"+ combinations+ ".h5")
            #    model.evaluate(x_val, y_val, verbose=1)
                # Predict on train, val and test
            #    preds_train = model.predict(x_train, verbose=1)
                preds_val = model.predict([x_test,np.reshape(y_test[:,:,:,0],newshape=(y_test.shape[0],y_test.shape[1],y_test.shape[2],1))], verbose=1)
                preds_val1 = preds_val[0]
                x_val_1 = np.reshape(y_test[:,:,:,0],newshape=(y_test.shape[1],y_test.shape[2],1))
#                y_val_1 = np.reshape(y_val[:,:,:,0],newshape=(y_test.shape[1],y_test.shape[1],1))
                pred_val_1 = preds_val1[:,:,:,:]
#                ndvi = y_val_1.astype('float32')
#                im = MODELS_PATH + city_name + '_' + preambol_name + '_' + str(patch0) + '_target__patches'+ str(size) +'_B' + combinations+'.tif'
#                imsave(im, ndvi)
                
                ndvi2 = pred_val_1.astype('float32')
                im2 =MODELS_PATH + city_name + '_' + preambol_name + '_' + str(patch0) +'_output_patches'+ str(size) +'_B' +combinations+'.tif'
                imsave(im2, ndvi2)
                
                ndvi3 = x_val_1.astype('float32')
                im3 = MODELS_PATH +city_name + '_' + preambol_name + '_' + str(patch0) +'_input_patches'+ str(size) +'_B' +combinations+'.tif'
                imsave(im3, ndvi3)
                
                k += 1
