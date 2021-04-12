# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from segmentation_models import  shallow_CNN#, shallow_CNN3#,shallow_CNN_2, shallow_CNN_3 #,Unet,PSPNet,Linknet,FPN
#from segmentation_models.backbones import get_preprocessing
#from segmentation_models.losses import bce_jaccard_loss
from keras.losses import mean_absolute_error
from keras.optimizers import Adam
#from segmentation_models.metrics import iou_score
from segmentation_models.load_data import load_data_super_resolution,combinations_input_super_resolution
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,Callback
from keras.layers import Conv2D, Input
from keras.models import Model
import time
import numpy as np
from tifffile import imsave
from mat4py import savemat, loadmat
import pickle
import os 
import argparse
from keras import backend as K


#Default values

batchs = (16,32,64,128)
nepochs = (1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
patchs = (17,32, 33, 64, 128)
nbands = (1,5,6,10)
ker1 = (1,3,5,9)
ker2 = (1,3,5,9)
ker3 = (1,3,5,9)
ker4 = (1,3,5,9)


parser = argparse.ArgumentParser(description = "Fast-Super-Resolution-20m-Sentinel-2-using-CNN")
parser.add_argument('--batch_size', required = True, choices = batchs, help="Specify the source language!")
parser.add_argument('--num_epochs', required = True, choices = nepochs, help="Specify the destination language!")
parser.add_argument('--patch_size', required = True, choices = patchs, help="Specify the destination language!")
parser.add_argument('--num_bands', required = True, choices = nbands, help="Specify the destination language!")
parser.add_argument('--kernel_1_size', required = True, choices = ker1, help="Specify the destination language!")
parser.add_argument('--kernel_2_size', required = True, choices = ker2, help="Specify the destination language!")
parser.add_argument('--kernel_3_size', required = True, choices = ker3, help="Specify the destination language!")
parser.add_argument('--kernel_4_size', required = True, choices = ker4, help="Specify the destination language!")

args = parser.parse_args()

#def tri_loss(y_true, y_pred):
#    x = y_true
#    y = y_pred
#    l1 = K.abs(x-y)
#    l2 = K.
##    mx = K.mean(x)
##    my = K.mean(y)
##    xm, ym = x-mx, y-my
##    r_num = K.sum(tf.multiply(xm,ym))
##    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
##    r = r_num / r_den
##
##    r = K.maximum(K.minimum(r, 1.0), -1.0)
##    mae = K.mean(K.abs(x - y))
#    maee = K.l2_normalize(K.abs(x - y))
#    return maee #(1 - r)# K.square(r)) #mae + 10*    

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


#indo = loadmat(r"D:\Dataset_SuperRisoluzione\indices.mat")
##indices = _['A']
#indices = 0
#time_callback = TimeHistory()

indices = 0    
# load your data
folder_1 = r"D:\Works\New folder1\\"
MODELS_PATH = r"C:\Users\massi\Downloads\segmentation_models-master\docs\model_path\\"

ts = str(int(time.time())) #Time stamp for the model distinctness
model_path = os.path.join(MODELS_PATH, 'model-{}.h5'.format(ts))
history_path = os.path.join(MODELS_PATH, 'model-{}.history'.format(ts))

#size = 33#128
#n_epochs = 20
#n_batch = 64
#
#num_bands = 10
#k_1 = 3#9
#k_2 = 3#5
#k_3 = 3#5
#bande = ["05","06","07","8A","11","12"]
#k = 0

model_name = ["shallow_Conc", "shallow"]
for mm in model_name: 
    size = 33#128
    n_epochs = 20
    n_batch = 64
    
    num_bands = 10
    k_1 = 3#9
    k_2 = 3#5
    k_3 = 3#5
    bande = ["05","06","07","8A","11","12"]
    k = 0
    
    
    for combinations in bande:
        x_train, y_train, x_val, y_val = load_data_super_resolution(folder_1, size,combinations,num_bands)#,indices)
    #    x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = combinations_input_super_resolution(x_traina, y_traina, x_vala, y_vala,x_train2a, y_train2a, x_val2a, y_val2a, comb[combinations],num[combinations] )
        np.savez("train_data.npz", x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val)    
        train_val = np.load("train_data.npz")
        x_train = train_val['x_train']
        y_train = train_val['y_train']
        x_val = train_val['x_val']
        y_val = train_val['y_val']
        N = x_train.shape[-1]
        print(x_train.shape)
        print(x_val.shape)
        if mm == "shallow_Conc":
            model = shallow_CNN(num_bands, k_1, k_2, k_3)
        else: 
            model = shallow_CNN(num_bands, k_1, k_2, k_3)
        Adamax = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile( loss=mean_absolute_error, metrics=[mean_absolute_error], optimizer=Adamax)
        #'Adam',
        
        ## fit model
        history2 = LossHistory()
        
        callbacks = [
            EarlyStopping(patience=10, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
            ModelCheckpoint("temporary_model_"+ combinations+ "_" + mm + ".h5", verbose=1, save_best_only=True, save_weights_only=True),
#            time_callback,
            history2
        ]
        
        history = model.fit(
            x=[x_train,np.reshape(y_train[:,:,:,1],newshape=(y_train.shape[0],y_train.shape[1],y_train.shape[2],1))], #reverse dimensions (in load_data_SR) and input from y_train
            y=np.reshape(y_train[:,:,:,0],newshape=(y_train.shape[0],y_train.shape[1],y_train.shape[2],1)),
            batch_size=n_batch,
            epochs=n_epochs,
            validation_data=([x_val,np.reshape(y_val[:,:,:,1],newshape=(y_val.shape[0],y_val.shape[1],y_val.shape[2],1))],np.reshape(y_val[:,:,:,0],newshape=(y_val.shape[0],y_val.shape[1],y_val.shape[2],1))),callbacks = callbacks
        )
    
    #    pickle.dump(history.history, open(history_path, "wb"))
    #    print('Saved model at {}'.format(model_path))
    #    print('Saved model history at {}'.format(history_path))
    #    dic = {}
    #    for a in history.history.keys():
    #        dic[a] =  np.asarray(history.history[a])
    #    dic_loss = {}
    #    dic_loss['losses'] = history2.losses
    #    savemat(os.path.join(MODELS_PATH, 'loss-{}_'.format(ts)+'_B'+ combinations+'.mat'),dic_loss)
    #
        times = time_callback.times
        dic_times = {}
        dic_times['times'] = times
        savemat('times.mat', dic_times)
        model.save(MODELS_PATH + "model_"+ combinations+ "_" + mm + ".h5")
    
        model.save_weights(MODELS_PATH + "model_weigths_"+ combinations+ "_" + mm + ".h5")
        
        
        # Load best model
        model.load_weights(MODELS_PATH + "model_weigths_"+ combinations+ "_" + mm + ".h5")
        
    #    model.evaluate(x_val, y_val, verbose=1)
        # Predict on train, val and test
    #    preds_train = model.predict(x_train, verbose=1)
        preds_val = model.predict([x_val,np.reshape(y_val[:,:,:,1],newshape=(y_val.shape[0],y_val.shape[1],y_val.shape[2],1))], verbose=1)
        
        for kk in range(8): 
            x_val_1 = np.reshape(y_val[50*kk,:,:,1],newshape=(y_val.shape[1],y_val.shape[1],1))
            y_val_1 = np.reshape(y_val[50*kk,:,:,0],newshape=(y_val.shape[1],y_val.shape[1],1))
            pred_val_1 = preds_val[50*kk,:,:,:]
            print(x_val_1.shape)
            print(y_val_1.shape)
            print(pred_val_1.shape)
            ndvi = y_val_1.astype('float32')
            im = MODELS_PATH +'target_'+ str(kk) + '_patches'+ str(size) +'_B' + combinations+ "_" + mm +'.tif'
            imsave(im, ndvi)
            
            ndvi2 = pred_val_1.astype('float32')
            im2 =MODELS_PATH +'output_'+ str(kk) + '_patches'+ str(size) +'_B' +combinations+ "_" + mm +'.tif'
            imsave(im2, ndvi2)
            
            ndvi3 = x_val_1.astype('float32')
            im3 = MODELS_PATH +'input_'+ str(kk) + '_patches'+ str(size) +'_B' +combinations+ "_" + mm +'.tif'
            imsave(im3, ndvi3)
        
        k += 1
