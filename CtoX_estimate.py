# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from segmentation_models import Unet,PSPNet,Linknet,FPN
from segmentation_models.backbones import get_preprocessing
from keras.optimizers import Adam

from segmentation_models import shallow_CNN#,shallow_CNN#,Unet,PSPNet,Linknet,FPN
#from segmentation_models.backbones import get_preprocessing
#from segmentation_models.losses import bce_jaccard_loss
from keras.losses import mean_absolute_error,kullback_leibler_divergence
from segmentation_models.losses import bce_jaccard_loss,cce_jaccard_loss
from keras.optimizers import Adam
#from segmentation_models.metrics import iou_score
from segmentation_models.load_data import load_data_tandem_x, combinations_input_super_resolution
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,Callback
from keras.layers import Conv2D, Input, BatchNormalization, Add
from keras.models import Model
import time
import numpy as np
from tifffile import imsave
from mat4py import savemat, loadmat
import pickle
import os 
import math


from keras import backend as K
import tensorflow as tf 
def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
#    mx = K.mean(x)
#    my = K.mean(y)
#    xm, ym = x-mx, y-my
#    r_num = K.sum(tf.multiply(xm,ym))
#    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
#    r = r_num / r_den
#
#    r = K.maximum(K.minimum(r, 1.0), -1.0)
#    mae = K.mean(K.abs(x - y))
    maee = K.l2_normalize(K.abs(x - y))
    return maee #(1 - r)# K.square(r)) #mae + 10*

#def PSNR_loss(y_true, y_pred):
#    max_pixel = K.max(y_true)
#    return 10.0 * math.log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) 

#from scipy.io import sio
#import argparse

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = np.max(img1)
    return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))


def psnr_loss(img1, img2):
    mse = K.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = K.max(img1)
    mae = K.mean(K.abs(img1 - img2))
    return  mae + 1000*K.mean(K.abs(K.log(img1/ img2)))

def my_mae_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    mae = K.sum(K.square(xm)) - K.sum(K.square(ym))

    return mae #(1 - r)# K.square(r)) #mae + 10*

def my_mae_wo_mean_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    mae1 = K.sum(K.abs(x - y)) # K.sum(K.square(K.abs(x) - K.abs(y))) #  #+ 
    mae2 = 1 - K.sum(K.square(ym))/K.sum(K.square(xm)) # K.sum(K.square(ym)) # 
    mae3 = 1 - K.sum(y)/K.sum(x)
    mae = mae1 + mae2 + mae3
    return mae #(1 - r)# K.square(r)) #mae + 10*



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

#from others import parser_xml
#xml_file=parser_xml(r"C:\Users\massi\Downloads\segmentation_models-master\config.xml")


#indo = loadmat(r"D:\Dataset_SuperRisoluzione\indices.mat")
#indices = _['A']
indices = 0
time_callback = TimeHistory()

    
# load your data
#folder_1 = r"D:\Works\DLR\Uganda_Solberg\\"
#MODELS_PATH = r"D:\Works\DLR\Uganda_Solberg\model_path\\"

folder_1 = r"D:\Works\DLR\Landcover\\"
MODELS_PATH = r"D:\Works\DLR\Landcover\model_path\\"


ts = str(int(time.time())) #Time stamp for the model distinctness
model_path = os.path.join(MODELS_PATH, 'model-{}.h5'.format(ts))
history_path = os.path.join(MODELS_PATH, 'model-{}.history'.format(ts))

size = 128#64#
n_epochs = 500#200 #
n_batch = 16 #16
model_name = "_CtoX" #
num_bands = 2#1
k_1 = 3# 9#
k_2 = 3#5#
k_3 = 3#5#
l_rate = 0.05 #5 #0.05
#bande = ["05","06","07","8A","11","12"]
#k = 0
#for combinations in bande:
#x_train, y_train, x_val, y_val = load_data_tandem_x(folder_1, size,num_bands,indices)

#np.savez("train_data.npz", x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val)    
train_val = np.load("train_data.npz")

x_train1 = train_val['x_train']
y_train1 = train_val['y_train']
x_val1 = train_val['x_val']
y_val = train_val['y_val']
x_train = x_train1[:n_batch*1600, :,:,:2]
#x_train = np.reshape(x_train[:, :,:,0],(x_train.shape[0], x_train.shape[1],x_train.shape[2], 1))
y_train = y_train1[:n_batch*1600, :,:,:]
x_val = x_val1[:, :,:,:2]

#x_val = np.reshape(x_val1[:, :,:,0],(x_val1.shape[0], x_val1.shape[1],x_val1.shape[2], 1))


#    x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = combinations_input_super_resolution(x_traina, y_traina, x_vala, y_vala,x_train2a, y_train2a, x_val2a, y_val2a, comb[combinations],num[combinations] )
N = x_train.shape[-1]
print(x_train.shape)
print(x_val.shape)
#train_ground = {"details": y_train[:,:,:,1],"band": y_train[:,:,:,0]}
#val_ground = {"details": y_val[:,:,:,1],"band": y_val[:,:,:,0]}

#model = shallow_CNN(num_bands , k_1, k_2, k_3)

#model.load_weights(MODELS_PATH + "model_weigths"+model_name+".h5")

BACKBONE = 'mobilenetv2'
#size1 = size//4
#size2 = 4*size
base_model = Unet(BACKBONE,input_shape=(size,size, 3), classes=1, activation='sigmoid', encoder_weights='imagenet', freeze_encoder=False)
inp = Input(shape=(size, size, 2))
##bn = BatchNormalization()(inp)
l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
outi = base_model(l1)
out2 = K.reshape(K.argmax(outi),(size,size,1))
outi = Add()([K.cast(out2,dtype="float32"), K.cast(inp,dtype="float32")])
#outi = base_model(inp)
#out = Add(name="band")([outi, inp])
model = Model(inp, outi, name=base_model.name)

#
Adamax = Adam(lr=l_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#####
#losses = {
#"details": "mean_absolute_error" ,# my_mae_wo_mean_loss , # my_mae_loss,# 
#"band": "mean_squared_error", #correlation_coefficient_loss , # #
#}
##
#lossWeights = {"details": 1.0, "band": 100.0 }
#metrics2 = {
#"details": "mean_absolute_error" ,
##my_mae_wo_mean_loss , # my_mae_loss,# "mean_absolute_error" ,
#"band": "mean_squared_error", 
##correlation_coefficient_loss , # "mean_squared_error",#my_mae_loss, #my_mae_loss, #
##	"band":,# "kullback_leibler_divergence",        
#        }
##model.compile( loss=losses, loss_weights=lossWeights, metrics=[mean_absolute_error], optimizer=Adamax)
#model.compile( loss=losses, loss_weights=lossWeights, metrics=metrics2, optimizer=Adamax)

#####

#model.compile( loss=my_mae_wo_mean_loss, metrics=[mean_absolute_error], optimizer=Adamax)
model.compile( loss=cce_jaccard_loss, metrics=[cce_jaccard_loss], optimizer=Adamax)
## fit model
history2 = LossHistory()

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=l_rate*0.001, verbose=1),
    ModelCheckpoint(MODELS_PATH + "temporary_model"+model_name+".h5", verbose=1, save_best_only=True, save_weights_only=True),
    time_callback,
    history2
]

history = model.fit(
    x= x_train, #[x_train, np.reshape(x_train[:,:,:,0],newshape=(y_train.shape[0],y_train.shape[1],y_train.shape[2],1))], #[x_train,np.reshape(y_train[:,:,:,1],newshape=(y_train.shape[0],y_train.shape[1],y_train.shape[2],1))], #reverse dimensions (in load_data_SR) and input from y_train # x_train,
    y= y_train, #[np.reshape(y_train[:,:,:,1], (x_train.shape[0], x_train.shape[1],x_train.shape[2], 1)), np.reshape(y_train[:,:,:,0], (x_train.shape[0], x_train.shape[1],x_train.shape[2], 1))], #y_train, #[np.reshape(y_train[:,:,:,0],newshape=(y_train.shape[0],y_train.shape[1],y_train.shape[2],1)),np.reshape(y_train[:,:,:,6],newshape=(y_train.shape[0],y_train.shape[1],y_train.shape[2],1))], #######  {"details": np.reshape(y_train[:,:,:,1], (x_train.shape[0], x_train.shape[1],x_train.shape[2], 1)),"band": np.reshape(y_train[:,:,:,0], (x_train.shape[0], x_train.shape[1],x_train.shape[2], 1))}, #y_train, #[np.reshape(y_train[:,:,:,0],newshape=(y_train.shape[0],y_train.shape[1],y_train.shape[2],1)),np.reshape(y_train[:,:,:,6],newshape=(y_train.shape[0],y_train.shape[1],y_train.shape[2],1))],
    batch_size=n_batch,
    epochs=n_epochs, #[x_val,np.reshape(y_val[:,:,:,1],newshape=(y_val.shape[0],y_val.shape[1],y_val.shape[2],1))]
    validation_data= (x_val, y_val), #([x_val, np.reshape(x_val[:,:,:,0],newshape=(x_val.shape[0],x_val.shape[1],x_val.shape[2],1))],[np.reshape(y_val[:,:,:,1],newshape=(y_val.shape[0],y_val.shape[1],y_val.shape[2],1)),np.reshape(y_val[:,:,:,0],newshape=(y_val.shape[0],y_val.shape[1],y_val.shape[2],1))]),callbacks = callbacks
#    validation_data=(x_val, np.reshape(y_val[:,:,:,0],(x_val.shape[0], x_val.shape[1],x_val.shape[2], 1))), callbacks = callbacks #y_val),callbacks = callbacks {"details": np.reshape(y_val[:,:,:,1],(x_val.shape[0], x_val.shape[1],x_val.shape[2], 1)),"band": np.reshape(y_val[:,:,:,0],(x_val.shape[0], x_val.shape[1],x_val.shape[2], 1))}), callbacks = callbacks #y_val),callbacks = callbacks
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
savemat(MODELS_PATH + 'times'+model_name+'.mat', dic_times)
model.save(MODELS_PATH + "model"+model_name+".h5")

model.save_weights(MODELS_PATH + "model_weigths"+model_name+".h5")

print(np.max(y_val))
# Load best model
#model.load_weights(MODELS_PATH + "model_weigths"+model_name+".h5")
#
##    model.evaluate(x_val, y_val, verbose=1)
## Predict on train, val and test
##    preds_train = model.predict(x_train, verbose=1)
#preds_val = model.predict(x_val, verbose=1)
#k = np.abs(y_val - preds_val)
print(np.max(y_val))
print(np.mean(y_val))
#print(np.mean(np.abs(y_val - x_val)))
#print(np.max(np.abs(y_val - x_val)))
#print(np.mean(k))
#print(np.max(k))
#for kk in range(8): 
#    y_val_1 = np.reshape(y_val[50*kk,:,:,0],newshape=(y_val.shape[1],y_val.shape[2],1))
#    x_val_1 = np.reshape(x_val[50*kk,:,:,0],newshape=(y_val.shape[1],y_val.shape[2],1))
#    pred_val_1 = preds_val[50*kk,:,:,:]
#    print(y_val_1.shape)
#    print(pred_val_1.shape)
#    ndvi = y_val_1.astype('float32')
#    im = MODELS_PATH +'target_XSAR_' + str(kk)+ model_name + '.tif'
#    imsave(im, ndvi)
#    
#    ndvi2 = pred_val_1.astype('float32')
#    im2 =MODELS_PATH +'output_XSAR_' + str(kk)+ model_name + '.tif'
#    imsave(im2, ndvi2)
#    
#    ndvi3 = x_val_1.astype('float32')
#    im3 =MODELS_PATH +'input_XSAR_' + str(kk)+ model_name +'.tif'
#    imsave(im3, ndvi3)


error = []
CC = []
psnr1 = []
errr = []
std = []
error2 = []
CC2 = []
psnr12 = []
errr2 = []
std2 = []

model.load_weights(MODELS_PATH + "model_weigths"+model_name+".h5")
preds_val = model.predict([x_val, np.reshape(x_val[:,:,:,0],newshape=(x_val.shape[0],x_val.shape[1],x_val.shape[2],1))], verbose=1)
preds_val = preds_val[1]

for kk in range(10):#range(0,y_val.shape[0],200): 
    y_val_1 = np.reshape(y_val[kk,:,:,0],newshape=(y_val.shape[1],y_val.shape[2]))
    land_1 = np.reshape(y_val[kk,:,:,2],newshape=(y_val.shape[1],y_val.shape[2]))
    x_val_1 = np.reshape(x_val[kk,:,:,0],newshape=(y_val.shape[1],y_val.shape[2]))
    pred_val_1 = preds_val[kk,:,:,:]
    pred_val_1 = np.reshape(preds_val[kk,:,:,:],newshape=(y_val.shape[1],y_val.shape[2]))
    ndvi2 = pred_val_1.astype('float32')
    im2 =MODELS_PATH +'output_XSAR_' + model_name + '_' + str(kk)+ '.tif'
    imsave(im2, ndvi2)
    
    ndvi2 = x_val_1.astype('float32')
    im2 =MODELS_PATH +'input_NASADEM_' + model_name + '_'  + str(kk)+ '.tif'
    imsave(im2, ndvi2)

    ndvi2 = y_val_1.astype('float32')
    im2 =MODELS_PATH +'reference_XSAR_' + model_name + '_' + str(kk)+ '.tif'
    imsave(im2, ndvi2)

    ndvi2 = land_1.astype('float32')
    im2 =MODELS_PATH +'landsat_treecover_' + model_name + '_'  + str(kk)+ '.tif'
    imsave(im2, ndvi2)

    
    print(y_val_1.shape)
    print(pred_val_1.shape)
    Diff_noabs = pred_val_1 - y_val_1
    Diff = np.abs(pred_val_1 - y_val_1)
    CC.append(np.corrcoef(y_val_1, pred_val_1))
    psnr1.append(psnr(y_val_1, pred_val_1))
    error.append(np.mean(Diff))
    errr.append(np.mean(Diff_noabs))
    std.append(np.std(Diff_noabs))

    Diff_noabs2 = x_val_1 - y_val_1
    Diff2 = np.abs(x_val_1 - y_val_1)
    CC2.append(np.corrcoef(y_val_1, x_val_1))
    psnr12.append(psnr(y_val_1, x_val_1))
    error2.append(np.mean(Diff2))
    errr2.append(np.mean(Diff_noabs2))
    std2.append(np.std(Diff_noabs2))
    
print('MAE pred: ' + str(np.mean(error)))
print('PSNR pred : ' +  str(np.mean(psnr1)))
print('CC pred : ' +  str(np.mean(CC)))
print('mean and dev std pred : ' +  str(np.mean(errr)) + ' and ' + str(np.mean(std)))

print('MAE inp: ' + str(np.mean(error2)))
print('PSNR inp: ' +  str(np.mean(psnr12)))
print('CC inp: ' +  str(np.mean(CC2)))
print('mean and dev std pred : ' +  str(np.mean(errr2)) + ' and ' + str(np.mean(std2)))

#    ndvi = y_val_1.astype('float32')
#    im = MODELS_PATH +'target_XSAR_' + str(kk)+ model_name + '.tif'
#    imsave(im, ndvi)
#    
#    ndvi2 = pred_val_1.astype('float32')
#    im2 =MODELS_PATH +'output_XSAR_' + str(kk)+ model_name + '.tif'
#    imsave(im2, ndvi2)
#    
#    ndvi3 = x_val_1.astype('float32')
#    im3 =MODELS_PATH +'input_XSAR_' + str(kk)+ model_name +'.tif'
#    imsave(im3, ndvi3)
