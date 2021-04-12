# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import keras
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
from util import unsupervised_train_generator_vv2, unsupervised_val_generator_vv2,unsupervised_train_generator_vv, unsupervised_val_generator_vv,train_generator_3, val_generator_3, train_ml_models
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

# def cropize(img,PS):
#     w,h,d = img.shape
#     w = PS*(w//PS)
#     h = PS*(h//PS)
#     img = img[:w,:h,:]
# return img.reshape(h//PS, PS, w//PS, PS,d).swapaxes(1, 2).reshape(-1, PS, PS, d)

# def custom_loss(y_true, y_pred):
#              loss1=bce_jaccard_loss(y_true,y_pred)
#              loss2=center_loss(y_true,fc)
#              return loss1+lambda*loss2

def l_mae(y_true,y_pred):
    # if 
    return K.mean(K.abs( y_pred -y_true ))#**2)
def l_iou(y_true, y_pred):
    intersection = y_true * y_pred

    notTrue = 1 - y_true
    union = y_true + (notTrue * y_pred)

    return 1 - K.sum(intersection)/K.sum(union)
def l_ccentropy(y_true, y_pred): 
    LCC = -K.mean(y_true*K.log(y_pred))
    return LCC 
def l_iou2(y_true, y_pred): 
    return l_iou(y_true, y_pred) + l_ccentropy(y_true, y_pred) #bce_jaccard_loss(y_true, y_pred)

def l_iou3(y_true, y_pred): 
    return l_iou(y_true, y_pred) + categorical_crossentropy(y_true,y_pred) #bce_jaccard_loss(y_true, y_pred)

# def l_struc(y_true,y_pred):
    
#     K.set_epsilon(10**(-12))
#     Weights=np.ndarray(shape=(2,2,3,4),dtype='float32')

#     Weights[:,:,0,0]=np.asarray([[1, 0],[-1, 0]],dtype='float32')
#     Weights[:,:,0,1]=np.asarray([[1, -1],[0, 0]],dtype='float32')
#     Weights[:,:,0,2]=np.asarray([[1, 0],[0 ,-1]],dtype='float32')
#     Weights[:,:,0,3]=np.asarray([[0 ,1],[-1, 0]],dtype='float32')
#     Weights[:,:,1,0]=np.asarray([[1, 0],[-1, 0]],dtype='float32')
#     Weights[:,:,1,1]=np.asarray([[1, -1],[0, 0]],dtype='float32')
#     Weights[:,:,1,2]=np.asarray([[1, 0],[0 ,-1]],dtype='float32')
#     Weights[:,:,1,3]=np.asarray([[0 ,1],[-1, 0]],dtype='float32')
#     Weights[:,:,2,0]=np.asarray([[1, 0],[-1, 0]],dtype='float32')
#     Weights[:,:,2,1]=np.asarray([[1, -1],[0, 0]],dtype='float32')
#     Weights[:,:,2,2]=np.asarray([[1, 0],[0 ,-1]],dtype='float32')
#     Weights[:,:,2,3]=np.asarray([[0 ,1],[-1, 0]],dtype='float32')

#     grads = K.conv2d(y_pred - y_true, Weights)
#     return K.mean(K.sqrt(K.abs(grads + K.epsilon())))


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
# folder_1 = r"D:\Works\Albufera-SemanticSegmentation\S2_S1\\" #r"D:\Albufera\\"#

# ## testing a parte i
# folder_test = folder_1 #r"D:\Works\Albufera-SemanticSegmentation\Testing\S2_S1\\"
size = 128
size_t = 144 # 128 #
n_epochs = 10# 5# 1 #  3# 10# 1 #5 # 1 ## 20
n_batch = 32
LR_MAX =0.01 #0.05 #    0.005 # 0.02 # 
comb = ["TriVV", "TriVH"] # ["VV","VH", "VVaVH"]#
# num = [1,1,2]#,3,3,3,5]#1,#[2]# 
num = { "VV": 1, "VH": 1, "VVaVH":2, "TriVV":3,"TriVH":3, "Tri":6, "Tri_one": 6}# 6}# 
#class_weight = {0: 50., 1: 1., 2: 1.}
# class_weight = [40.0, 1.0, 1.0]

# x_traina, y_traina, x_vala, y_vala,x_train2a, y_train2a, x_val2a, y_val2a = load_data(folder_1, size,size_t,indo,indo1)
# print(x_traina.shape)
# print(x_vala.shape)
# np.savez("train_MetriAgriFOR.npz",x_traina = x_traina, y_traina = y_traina, x_vala = x_vala, y_vala = y_vala,x_train2a = x_train2a, y_train2a = y_train2a, x_val2a = x_val2a, y_val2a = y_val2a) #,  x_gtrain = x_gtrai y_train = y_trai  x_gval = x_gval, y_val = y_val)    

# train_MetriAgriFOR = np.load("train_MetriAgriFOR.npz")
dic = {}
for date_alb in range(0,1): #(4): #
    date_2 = str(1 )
    Out_classes = 3# 5
    date_3 = str(date_alb + 1)
    # train_MetriAgriFOR = np.load("train_MetriAgriFOR_"+ date_3 + "Date.npz")

    # # train_MetriAgriFOR_singleDate
    # x_traina = train_MetriAgriFOR["x_traina"]  
    # y_traina = train_MetriAgriFOR["y_traina"] 
    # x_vala = train_MetriAgriFOR["x_vala"]   
    # y_vala = train_MetriAgriFOR["y_vala"]  
    # x_train2a = train_MetriAgriFOR["x_train2a"]  
    # y_train2a = train_MetriAgriFOR["y_train2a"]   
    # x_val2a = train_MetriAgriFOR["x_val2a"] 
    # y_val2a = train_MetriAgriFOR["y_val2a"]
    # # test_MetriAgriFOR_with_summer_augmentation = np.load("test_MetriAgriFOR_with_summer_augmentation.npz")
    # # x_test = test_MetriAgriFOR_with_summer_augmentation["x_gtest"] 
    # # y_test = test_MetriAgriFOR_with_summer_augmentation["y_test"]#, x_gtrain = x_gtrain, y_train = y_train,  x_gval = x_gval, y_val = y_val)  
    #  
    files_test = [r"D:\fiumiunsupervised\Training_Set_Unsupervised_Borgoforte_withOpt\test_intradate1_beta"+date_3+".npz"]#, "test_intradate_gamma.npz", "test_intradate_beta.npz"]
    dic_name_inputs = {r"D:\fiumiunsupervised\Training_Set_Unsupervised_Borgoforte_withOpt\test_intradate1_beta"+date_3+".npz":"B11_8_4_", "test_intradate_gamma.npz":"gamma_", "test_intradate_a.npz":"beta_"}
    folder_trains = {r"D:\fiumiunsupervised\Training_Set_Unsupervised_Borgoforte_withOpt\test_intradate1_beta"+date_3+".npz":r"D:\fiumiunsupervised\Training_Set_Unsupervised_Borgoforte_withOpt\\", "test_intradate_gamma.npz":r"D:\Albufera_2019_processed\Gamma2\\", "test_intradate_a.npz":r"D:\Albufera_2019_processed\Beta2\\"}
    
    # files_test = ["test_intradate_beta.npz"]#, "test_intradate_gamma.npz", "test_intradate_beta.npz"]
    # dic_name_inputs = {"test_intradate_beta.npz":"sigma_", "test_intradate_gamma.npz":"gamma_", "test_intradate_a.npz":"beta_"}
    # folder_trains = {"test_intradate_beta.npz":r"D:\Albufera_2019_processed\Sigma2\\", "test_intradate_gamma.npz":r"D:\Albufera_2019_processed\Gamma2\\", "test_intradate_a.npz":r"D:\Albufera_2019_processed\Beta2\\"}
    
    for file_test in files_test: 
        folder_train =folder_trains[file_test] #r"D:\Albufera_2019_processed\Sigma2\\"
        dir_list = os.listdir(folder_train)
        dir_list.sort()

        number_train = 0
        number_val = 0


        for file_1 in dir_list: 
            if file_1.find("X_train_") != -1 and file_1.find(".npy") != -1:
                number_train += 1
            if file_1.find("X_val_") != -1 and file_1.find(".npy") != -1:
                number_val += 1

        ecc = np.arange(number_val)
########## file test 
        test_MetriAgriFOR_with_summer_augmentation = np.load(file_test)
        name_inputs =  dic_name_inputs[file_test]
        x_test = test_MetriAgriFOR_with_summer_augmentation["x_test"] 
        y_test = test_MetriAgriFOR_with_summer_augmentation["y_test"]#, x_gtrain = x_gtrain, y_train = y_train,  x_gval = x_gval, y_val = y_val)    

        for combinations in comb:
            # X, y = train_ml_models(folder_train, number_train, combinations,size)

            timer = time.time()
            N = num[combinations]
            # x_train, y_train, x_val, y_val,x_train2, y_train2, x_val2, y_val2 = combinations_input(x_traina, y_traina, x_vala, y_vala,x_train2a, y_train2a, x_val2a, y_val2a, combinations,num[combinations] )
            # N = x_train.shape[-1]
            # print(x_train.shape)
            # print(x_val.shape)
            backs = ['mobilenetv2']# ['resnet34','mobilenetv2']
            for k_back in range(len(backs)):
                activate3 = 'softmax' #     'linear' #
                BACKBONE = backs[k_back] # 
                preprocess_input = get_preprocessing(BACKBONE)        ## define model and chose between following models: 
                
                networks = ["Tri_Net"] #["Fractal_Net"] # ["DenseUNet"]# ["SegNet","FPN"]#   "NestNet" "shallow_CNN"] #  ["SVM", "RF", "GBC", 
                for k_mod in networks:
                #    size = size_t[k_mod]
                    name_model_pre  = k_mod+"_New"#+ "_despeck3_"# +"3"#
                    name_model = k_mod+"_New3"#+ "_despeck3_"# +"3"#

                    print(combinations + "_" + k_mod)
                    tensorboard = TensorBoard(log_dir='logs/{}'.format( name_inputs + combinations + "_" + k_mod))

                    # if k_mod == "ResNeXt50":
                    #     if N == 3: 
                    #         model = ResNeXt50(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True)
                    #     else:
                    #         base_model = ResNeXt50(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True)
                    #         inp = Input(shape=(size, size, N))
                    #         bn = BatchNormalization()(inp)
                    #         l1 = Conv2D(3, (1, 1))(bn) # map N channels data to 3 channels
                    #         out = base_model(l1)
                    #         model = Model(inp, out, name=base_model.name)
                    if k_mod == "SVM":
                        from sklearn.svm import SVC
                        print ('[INFO] Training Support Vector Machine model.')
                        model = SVC()
                        # model = Parallel(n_jobs=6)(delayed(model.fit)(X[i:i+100,:],y[i:i+100,:]) for i in range(0,100,X.shape[0]))
                        model.fit(X, y)
                    elif k_mod == "RF":
                        from sklearn.ensemble import RandomForestClassifier
                        print ('[INFO] Training Random Forest model.')
                        model = RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42,verbose=3,n_jobs=-1)
                        model.fit(X, y)
                    elif k_mod == "GBC":
                        from sklearn.ensemble import GradientBoostingClassifier
                        model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0,verbose=3,n_jobs=-1)
                        model.fit(X, y)
                    elif k_mod == "Unet":
                        if N == 3: 
                            model = Unet(BACKBONE,input_shape=(size,size, 3), classes=Out_classes, activation=activate3, encoder_weights='imagenet', freeze_encoder=False)
                        else:

                            base_model = Unet(BACKBONE,input_shape=(size,size, 3), classes=Out_classes, activation=activate3, encoder_weights='imagenet', freeze_encoder=False)
                            inp = Input(shape=(size, size, N))
                            bn = BatchNormalization()(inp)
                            l1 = Conv2D(3, (1, 1))(bn) # map N channels data to 3 channels
                            out = base_model(l1)
                            model = Model(inp, out, name=base_model.name)
                        
        #            elif k_mod == 1:
        #                N = x_train.shape[-1]
        #                if N == 3: 
        #                    model = PSPNet(BACKBONE, input_shape=(size_t, size_t, 3), classes=3, activation=activate3, encoder_weights='imagenet', encoder_freeze=False)
        #                else:
        #                    base_model = PSPNet(BACKBONE, input_shape=(size_t, size_t, 3), classes=3, activation=activate3, encoder_weights='imagenet', encoder_freeze=False)
        #                    inp = Input(shape=(size_t, size_t, N))
        #                    bn = BatchNormalization()(inp)
        #                    l1 = Conv2D(3, (1, 1))(bn) # map N channels data to 3 channels
        #                    out = base_model(l1)
        #                    model = Model(inp, out, name=base_model.name)
                        
                    elif k_mod == "Linknet":
                        # N = x_train.shape[-1]
                        if N == 3: 
                            model = Linknet(BACKBONE, input_shape=(size, size, 3), classes=Out_classes, activation=activate3, encoder_weights='imagenet',encoder_freeze=False)
                        else:
                            base_model = Linknet(BACKBONE, input_shape=(size, size, 3), classes=Out_classes, activation=activate3, encoder_weights='imagenet',encoder_freeze=False)
                            inp = Input(shape=(size, size, N))
                            bn = BatchNormalization()(inp)
                            l1 = Conv2D(3, (1, 1))(bn) # map N channels data to 3 channels
                            out = base_model(l1)
                            model = Model(inp, out, name=base_model.name)
                        
                    elif k_mod == "FPN":
                        # N = x_train.shape[-1]
                        if N == 3: 
                            model = FPN(BACKBONE, input_shape=(size, size, 3), classes=Out_classes, activation=activate3, encoder_weights='imagenet', encoder_freeze=False)
                        else:
                            base_model = FPN(BACKBONE, input_shape=(size, size, 3), classes=Out_classes, activation=activate3, encoder_weights='imagenet', encoder_freeze=False)
                            inp = Input(shape=(size, size, N))
                            bn = BatchNormalization()(inp)
                            l1 = Conv2D(3, (1, 1))(bn) # map N channels data to 3 channels
                            out = base_model(l1)
                            model = Model(inp, out, name=base_model.name)
                    elif k_mod == "NestNet":
                        # N = x_train.shape[-1]
                        if N == 3: 
                            

                            
                            model = Nestnet(backbone_name='vgg16', 
                input_shape=(size, size, 3),
                input_tensor=None,
                encoder_weights='imagenet',
                freeze_encoder=False,
                skip_connections='default',
                decoder_block_type='upsampling',
                decoder_filters=(64,32,16,8,4),
                decoder_use_batchnorm=True,
                n_upsample_blocks=5,
                upsample_rates=(2,2,2,2,2),
                classes=Out_classes,
                activation=activate3)
                        else:
                            base_model = Nestnet(backbone_name='vgg16', 
                input_shape=(size, size, 3),
                input_tensor=None,
                encoder_weights='imagenet',
                freeze_encoder=False,
                skip_connections='default',
                decoder_block_type='upsampling',
                decoder_filters=(64,32,16,8,4),
                decoder_use_batchnorm=True,
                n_upsample_blocks=5,
                upsample_rates=(2,2,2,2,2),
                classes=Out_classes,
                activation=activate3)
                            inp = Input(shape=(size, size, N))
                            bn = BatchNormalization()(inp)
                            l1 = Conv2D(3, (1, 1))(bn) # map N channels data to 3 channels
                            out = base_model(l1)
                            model = Model(inp, out, name=base_model.name)
                #    elif k_mod == "XNet":
                #        N = x_train.shape[-1]
                #        if N == 3: 
                #            model = Xnet(backbone_name='vgg16',
                # input_shape=(size, size, 3),
                # input_tensor=None,
                # encoder_weights='imagenet',
                # freeze_encoder=False,
                # skip_connections='default',
                # decoder_block_type='upsampling',
                # decoder_filters=(32,32,16,8,4),
                # decoder_use_batchnorm=True,
                # n_upsample_blocks=5,
                # upsample_rates=(2,2,2,2,2),
                # classes=3,
                # activation=activate3)
                #        else:
                #            base_model = Xnet(backbone_name='vgg16',
                # input_shape=(size, size, 3),
                # input_tensor=None,
                # encoder_weights='imagenet',
                # freeze_encoder=False,
                # skip_connections='default',
                # decoder_block_type='upsampling',
                # decoder_filters=(32,32,16,8,4),
                # decoder_use_batchnorm=True,
                # n_upsample_blocks=5,
                # upsample_rates=(2,2,2,2,2),
                # classes=3,
                # activation=activate3)
                #            inp = Input(shape=(size, size, N))
                #            bn = BatchNormalization()(inp)
                #            l1 = Conv2D(3, (1, 1))(bn) # map N channels data to 3 channels
                #            out = base_model(l1)
                #            model = Model(inp, out, name=base_model.name)
                    elif k_mod == "shallow_CNN":
                        # N = x_train.shape[-1]
                        active = 'relu'
                        active3 = activate3
                        inp = Input(shape=(None, None, N))
                        bn = BatchNormalization()(inp)
        #                l1 = Conv2D(64, kernel_size=3, activation= active, padding='same', kernel_initializer='he_normal' )(bn)
        #                bn1 = BatchNormalization()(l1)

                        l2 = Conv2D(64, kernel_size=3, activation=active, padding='same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(bn)
                        l3 = Conv2D(48, kernel_size=3, activation=active, padding='same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(l2)
                        l4 = Conv2D(32, kernel_size=3, activation=active, padding='same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(l3)
                        # out2 = Conv2D(2, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal', name= "L1")(l4)
                        # l5 = Conv2D(32, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(out2)
                        out = Conv2D(Out_classes, kernel_size=1, activation=active3, padding='same', kernel_initializer = keras.initializers.glorot_normal(seed=None), name="Class")(l4) #(l5)
                        
                    #    out= Conv2D(1, kernel_size=k_3, activation='relu', padding='same', kernel_initializer='he_normal',name="nothing")(out1)
                        model = Model(inp, out, name='shallow') # [out2, out], name='shallow')
                        # model = multi_gpu_model(model, 2)
                    elif k_mod == "shallow_CNN2":
                        # N = x_train.shape[-1]
                        active = 'relu'
                        active3 = activate3
                        inp = Input(shape=(None, None, N))
                        bn = BatchNormalization()(inp)
        #                l1 = Conv2D(64, kernel_size=3, activation= active, padding='same', kernel_initializer='he_normal' )(bn)
        #                bn1 = BatchNormalization()(l1)

                        l2 = Conv2D(64, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(bn)
                        l3 = Conv2D(48, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(l2)
                        l4b = Conv2D(32, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(l3)
                        # out2 = Conv2D(2, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal', name= "L1")(l4)
                        # l5 = Conv2D(32, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(out2)
                        l4 = concatenate([l2, l4b], axis=-1)
                        out = Conv2D(Out_classes, kernel_size=3, activation=active3, padding='same', kernel_initializer='he_normal', name="Class")(l4) #(l5)
                        
                    #    out= Conv2D(1, kernel_size=k_3, activation='relu', padding='same', kernel_initializer='he_normal',name="nothing")(out1)
                        model = Model(inp, out, name='shallow') # [out2, out], name='shallow')
                        # model = multi_gpu_model(model, 2)

                    elif k_mod == "shallow_CNN3":
                        # N = x_train.shape[-1]
                        active = 'relu'
                        active3 = activate3
                        inp = Input(shape=(None, None, N))
                        bn = BatchNormalization()(inp)
        #                l1 = Conv2D(64, kernel_size=3, activation= active, padding='same', kernel_initializer='he_normal' )(bn)
        #                bn1 = BatchNormalization()(l1)

                        l2 = Conv2D(64, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(bn)
                        l3b = Conv2D(48, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(l2)
                        l4b = Conv2D(32, kernel_size=3, activation=active, padding='same', kernel_initializer='he_normal')(l3)
                        l4c = concatenate([l2, l4b], axis=-1)
                        l4 = BatchNormalization()(l4c)
                        out = Conv2D(Out_classes, kernel_size=3, activation=active3, padding='same', kernel_initializer='he_normal', name="Class")(l4) #(l5)
                        
                    #    out= Conv2D(1, kernel_size=k_3, activation='relu', padding='same', kernel_initializer='he_normal',name="nothing")(out1)
                        model = Model(inp, out, name='shallow') # [out2, out], name='shallow')
                        # model = multi_gpu_model(model, 2)


                    elif k_mod == "Fractal_Net":           
                        f = 8
                        # N = x_train.shape[-1]
                        inputs = Input((size, size, N))
                        active = 'relu'
                        active_lin = 'linear'
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
                        conv99 = Conv2D(Out_classes, kernel_size=1, activation=activate3, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None), name="segmentation")(conv9)
                        # --- end first u block
                    
                        down1bb = MaxPooling2D(pool_size=(2, 2))(conv9)
                        down1bbb = MaxPooling2D(pool_size=(2, 2))(conv99)
                        # down1bb = MaxPooling2D(pool_size=(2, 2))(conv9)
                        down1b = concatenate([down1bb, down1bbb], axis=-1)
                        # down1b = merge([down1b, conv8], mode='concat', concat_axis=3)
                        up2 = concatenate([down1b, conv8], axis=-1)

                        conv2b = BatchNormalization()(up2) #down1b)
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
                        up1b = concatenate([UpSampling2D(size=(2, 2))(conv5b), down3b], axis=-1) #conv4b], axis=-1)


                        conv6b = BatchNormalization()(up1b)
                        # conv6b = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6b)
                        conv6b = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv6b)
                        conv6b = BatchNormalization()(conv6b)
                        # conv6b = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6b)
                        conv6b = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv6b)

                        # up2b = merge([UpSampling2D(size=(2, 2))(conv6b), conv3b], mode='concat', concat_axis=3)
                        up2b = concatenate([UpSampling2D(size=(2, 2))(conv6b), down2b], axis=-1) #conv3b], axis=-1)
                    
                        conv7b = BatchNormalization()(up2b)
                        # conv7b = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7b)
                        conv7b = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv7b)
                        conv7b = BatchNormalization()(conv7b)
                        # conv7b = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7b)
                        conv7b = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv7b)

                        # up3b = merge([UpSampling2D(size=(2, 2))(conv7b), conv2b], mode='concat', concat_axis=3)
                        up3b = concatenate([UpSampling2D(size=(2, 2))(conv7b), down1b], axis=-1) #conv2b], axis=-1)

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
                        outputs = Conv2D(N, kernel_size=1, activation=active_lin, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None), name="mae_output")(conv9b)
                        # outputs = Convolution2D(3, 1, 1, activation=activate3, border_mode='same')(conv9b)
                        # outputs = Conv2D(Out_classes, kernel_size=1, activation=activate3, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv9b)
                        model = Model(inputs=inputs, outputs=[conv99, outputs])  

                    elif k_mod == "Tri_Net":           
                        f = 8
                        # N = x_train.shape[-1]
                        inputs = Input((size, size, N))
                        active = 'relu'
                        active_lin = 'linear'
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
                        conv99 = Conv2D(Out_classes, kernel_size=1, activation=activate3, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None), name="segmentation")(conv9)
                        # --- end first u block
                    
                        down1bb = MaxPooling2D(pool_size=(2, 2))(conv9)
                        down1bbb = MaxPooling2D(pool_size=(2, 2))(conv99)
                        # down1bb = MaxPooling2D(pool_size=(2, 2))(conv9)
                        down1b = concatenate([down1bb, down1bbb], axis=-1)
                        # down1b = merge([down1b, conv8], mode='concat', concat_axis=3)
                        up2 = concatenate([down1b, conv8], axis=-1)

                        conv2b = BatchNormalization()(up2) #down1b)
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
                        up1b = concatenate([UpSampling2D(size=(2, 2))(conv5b), down3b], axis=-1) #conv4b], axis=-1)


                        conv6b = BatchNormalization()(up1b)
                        # conv6b = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6b)
                        conv6b = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv6b)
                        conv6b = BatchNormalization()(conv6b)
                        # conv6b = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6b)
                        conv6b = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv6b)

                        # up2b = merge([UpSampling2D(size=(2, 2))(conv6b), conv3b], mode='concat', concat_axis=3)
                        up2b = concatenate([UpSampling2D(size=(2, 2))(conv6b), down2b], axis=-1) #conv3b], axis=-1)
                    
                        conv7b = BatchNormalization()(up2b)
                        # conv7b = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7b)
                        conv7b = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv7b)
                        conv7b = BatchNormalization()(conv7b)
                        # conv7b = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7b)
                        conv7b = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv7b)

                        # up3b = merge([UpSampling2D(size=(2, 2))(conv7b), conv2b], mode='concat', concat_axis=3)
                        up3b = concatenate([UpSampling2D(size=(2, 2))(conv7b), down1b], axis=-1) #conv2b], axis=-1)

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
                        if combinations == "Tri_one":
                            outputs = Conv2D(2, kernel_size=1, activation=active_lin, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None), name="mae_output")(conv9b)
                        else: 
                            outputs = Conv2D(N, kernel_size=1, activation=active_lin, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None), name="mae_output")(conv9b)
                        # outputs = Conv2D(4, kernel_size=1, activation=active_lin, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None), name="mae_output")(conv9b)
                        # outputs = Convolution2D(3, 1, 1, activation=activate3, border_mode='same')(conv9b)
                        # outputs = Conv2D(Out_classes, kernel_size=1, activation=activate3, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv9b)
            
                        
                        ##### end second block 

                        conv1c = Conv2D(f, kernel_size=3, activation=active, padding='same', kernel_initializer = keras.initializers.glorot_normal(seed=None) )(outputs)
                        conv1c = BatchNormalization()(conv1c)
                        # conv1 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv1)
                        conv1c = Conv2D(f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv1c)

                        down1c = MaxPooling2D(pool_size=(2, 2))(conv1c)
                    
                        conv2c = BatchNormalization()(down1c)
                        # conv2 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv2)
                        conv2c = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv2c)
                        conv2c = BatchNormalization()(conv2c)
                        # conv2 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv2)
                        conv2c = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv2c)
                        
                        down2c = MaxPooling2D(pool_size=(2, 2))(conv2c)
                    
                        conv3c = BatchNormalization()(down2c)
                        # conv3 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv3)
                        conv3c = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv3c)
                        conv3c = BatchNormalization()(conv3c)
                        # conv3 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv3)
                        conv3c = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv3c)

                        down3c = MaxPooling2D(pool_size=(2, 2))(conv3c)
                    
                        conv4c = BatchNormalization()(down3c)
                        # conv4 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv4)
                        conv4c = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv4c)
                        conv4c = BatchNormalization()(conv4c)
                        # conv4 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv4)
                        conv4c = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv4c)


                        down4c = MaxPooling2D(pool_size=(2, 2))(conv4c)
                    
                        conv5c = BatchNormalization()(down4c)
                        # conv5 = Convolution2D(16 * f, 3, 3, activation='relu', border_mode='same')(conv5)
                        conv5c = Conv2D(16*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv5c)
                        conv5c = BatchNormalization()(conv5c)
                        # conv5 = Convolution2D(16 * f, 3, 3, activation='relu', border_mode='same')(conv5)
                        conv5c = Conv2D(16*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv5c)

                        # up1 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
                        up1c = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
                        conv6c = BatchNormalization()(up1c)
                        conv6c = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv6c)
                        # conv6 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6)
                        conv6c = BatchNormalization()(conv6c)
                        # conv6 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6)
                        conv6c = Conv2D(8*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv6c)


                        # up2 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
                        up2c = concatenate([UpSampling2D(size=(2, 2))(conv6c), conv3c], axis=-1)
                        conv7c = BatchNormalization()(up2c)
                        # conv7 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7)
                        conv7c = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv7c)
                        conv7c = BatchNormalization()(conv7c)
                        # conv7 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7)
                        conv7c = Conv2D(4*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv7c)

                        # up3 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
                        up3c = concatenate([UpSampling2D(size=(2, 2))(conv7c), conv2c], axis=-1)

                        conv8c = BatchNormalization()(up3c)
                        # conv8 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv8)
                        conv8c = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv8c)
                        conv8c = BatchNormalization()(conv8c)
                        # conv8 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv8)
                        conv8c = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv8c)

                        # up4 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
                        up4c = concatenate([UpSampling2D(size=(2, 2))(conv8c), conv1c], axis=-1)

                        conv9c = BatchNormalization()(up4c)
                        # conv9 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv9)
                        conv9c = Conv2D(f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv9c)
                        conv9c = BatchNormalization()(conv9c)
                        # conv9 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv9)
                        conv9c = Conv2D(f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv9c)
                        outputsc = Conv2D(Out_classes, kernel_size=1, activation=activate3, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None), name="segmentation_pp")(conv9c)
                        
                        model = Model(inputs=inputs, outputs=[conv99, outputs, outputsc])  




                    elif k_mod == "DenseUNet":           
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
                        conv8 = BatchNormalization()(up3)
                        # conv8 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv8)
                        conv8 = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv8)
                        conv8 = BatchNormalization()(conv8)
                        # conv8 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv8)
                        conv8 = Conv2D(2*f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv8)

                        # up4 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
                        up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
                        up4 = concatenate([UpSampling2D(size=(2, 2))(conv2), up4], axis=-1)
                        conv9 = BatchNormalization()(up4)
                        # conv9 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv9)
                        conv9 = Conv2D(f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv9)
                        conv9 = BatchNormalization()(conv9)
                        # conv9 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv9)
                        conv9 = Conv2D(f, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv9)
                    
                        # outputs = Convolution2D(3, 1, 1, activation=activate3, border_mode='same')(conv9b)
                        outputs = Conv2D(Out_classes, kernel_size=1, activation='softmax', padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None), name="segmentation")(conv9)
                        est_inputs = Conv2D(N, kernel_size=3, activation=active, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None), name="mae_output")(conv9)
                        model = Model(inputs=inputs, outputs=[ outputs,est_inputs])    
                    elif k_mod == "SegNet":
                        # N = x_train.shape[-1]
                        
                        model = SegNet.SegNet(Out_classes, input_height=size, input_width=size, input_bands=N, activate3 = activate3)

                    elif k_mod == "FCN8":
                        # N = x_train.shape[-1]
                        model = FCN8.FCN8(Out_classes, input_height=size, input_width=size, input_bands=N)
                    # print(k_mod)
                    # print(model.count_params())
                    dic[k_mod] = model.count_params()
# print(dic)
                    # print(model.summary())

            #                 # model = Model(inp, out, name=base_model.name)
            #         # model.load_weights( name_inputs +combinations + "_" + BACKBONE + "_" + name_model_pre + "_model_wIoU"+ str(size) +".h5")
            # ######## TRAINING ############
                    if k_mod == "SVM" or k_mod == "RF" or k_mod == "GBC" :
                        model.fit(X, y)
                    else: 
                        # if k_mod == "shallow_CNN":  
                        # model.load_weights(date_3 + name_inputs +combinations + "_Supervised" + BACKBONE + "_" + name_model + "_model_wIoU"+ str(size) +".h5")

                        Adamax = Adam(lr = LR_MAX, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                        ##### lr=0.0001


                        losses = {"segmentation_pp": l_iou, "segmentation": l_iou, "mae_output": l_mae} # categorical_crossentropy, "mae_output": l_mae}
                        # lossWeights = {"segmentation_pp": 0.79, "segmentation": 0.201, "mae_output":  0.009}
                        lossWeights = {"segmentation_pp": 0.7, "segmentation": 0.25, "mae_output":  0.05}
                        # lossWeights = {"segmentation": 1.0, "mae_output":  0.0}
                        model.compile(loss = losses, loss_weights = lossWeights, metrics = [l_iou,l_mae], optimizer = Adamax) # 
                        # dice_loss categorical_crossentropy  bce_jaccard_loss  cce_jaccard_loss cce_dice_loss , metrics=[iou_score], optimizer=Adamax)

                        # model.compile(loss=MAE, metrics = [MAE], optimizer = Adamax) # keras.losses.sparse_c
            #            model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
            #            
                        callbacks = [
                            EarlyStopping(patience=10, verbose=1),
                            ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
                            ModelCheckpoint( name_inputs + combinations + "_" + BACKBONE + "_" + name_model + "_temporary_model_IoU"+ str(size) +".h5", verbose=1, save_best_only=True, save_weights_only=True),
                            tensorboard
                            # time_callback
                        ]

                        print("number_train " + str(number_train))
                        number_train = number_train#//2
                        s_p_e = number_train//n_batch # x_train.shape[0]//n_batch # 
                        train_val_p2 = n_batch*s_p_e #x_train.shape[0] # n_batch*50#250# #n_batch*500 
                        val_p2 = n_batch*10
                        val_pe = val_p2//n_batch
                        print(type(s_p_e))

                        # model.fit_generator(unsupervised_train_generator_vv(train_val_p2, number_train, n_batch,folder_train, size,N, combinations), validation_data = unsupervised_val_generator_vv(train_val_p2, number_val, n_batch,folder_train, size,N, combinations), validation_steps = val_pe, steps_per_epoch = s_p_e, epochs = n_epochs,  callbacks = callbacks, class_weight=[[100.0,12.0, 2.5],[1.0],[100.0,12.0, 2.5]]) #200.0,100.0, 1.0
                        model.fit_generator(unsupervised_train_generator_vv2(train_val_p2, number_train, n_batch,folder_train, size,N, combinations), validation_data = unsupervised_val_generator_vv2(train_val_p2, number_val, n_batch,folder_train, size,N, combinations), validation_steps = val_pe, steps_per_epoch = s_p_e, epochs = n_epochs,  callbacks = callbacks, class_weight=[[200.0,50.0, 1.0],[1.0],[200.0,50.0, 1.0]]) #200.0,100.0, 1.0
                        # model.fit_generator(train_generator_3(train_val_p2, number_train, n_batch,folder_train, size,N, combinations, x_train, y_train), validation_data = val_generator_3(train_val_p2, number_val, n_batch,folder_train, size,N, combinations, x_train, y_train), validation_steps = val_pe, steps_per_epoch = s_p_e, epochs = n_epochs,callbacks = callbacks) #class_weight=[0.4,0.5, 0.1], callbacks = callbacks)

                        # # fit model
                        # if name_model == "PSPNet":
                        #     model.fit(
                        #         x=x_train2,
                        #         y=y_train2,
                        #         batch_size=n_batch,
                        #         epochs=n_epochs,
                        #         class_weight = class_weight,
                        #         validation_data=(x_val2, y_val2),callbacks = callbacks
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
                        times = time.time() - timer 
                        dic_times = {}
                        dic_times['times'] = times
                        savemat( name_inputs +combinations + "_" + BACKBONE + '_' + name_model + '_times.mat', dic_times)
                        model.save_weights( date_2 + name_inputs +combinations + "_Supervised_Post_TriVV" + BACKBONE + "_" + name_model + "_model_wIoU"+ str(size) +".h5")
       
       
        # ############END TRAINING#############           
                    


    # #########TESTING ############################
                        if name_model == "PSPNet":
                            preds_train = model.predict(x_train2, verbose=1)
                            preds_val = model.predict(x_val2, verbose=1)
                            for k in range(0,x_val.shape[0],int(x_val.shape[0]/100)):
                                x_val_1 = x_val2[k,:,:,:]
                                y_val_1 = y_val2[k,:,:,:]
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
                        else:
            #                preds_train = model.predict(x_train, verbose=1)
                            # dataset = gdal.Open(os.path.join(folder_test, str(1) + "_Albufera_VV.tif"), gdal.GA_ReadOnly)
                            # vv_size = dataset.ReadAsArray()
                            # dataset = None
                            # s1 = vv_size.shape[1]
                            # s2 = vv_size.shape[2]
                            # y_test = np.ndarray(shape=(5, s1, s2, 3), dtype='float32')
                            # x_test = np.ndarray(shape=(5, s1, s2, N), dtype='float32')
                            # for n_ in range(0,5):

                            #     dataset = gdal.Open(os.path.join(folder_test, str(n_+1)+"_Albufera_VV.tif"), gdal.GA_ReadOnly)
                            #     vv_1 = dataset.ReadAsArray()
                            #     dataset = None
                            #     dataset = gdal.Open(os.path.join(folder_test, str(n_+1)+"_Albufera_VH.tif"), gdal.GA_ReadOnly)
                            #     vh_1 = dataset.ReadAsArray()
                            #     dataset = None
                            #     dataset = gdal.Open(os.path.join(folder_test, str(n_+1)+"_vegetation.tif"), gdal.GA_ReadOnly)
                            #     veg_1 = dataset.ReadAsArray()
                            #     dataset = None
                            #     dataset = gdal.Open(os.path.join(folder_test, str(n_+1)+"_water.tif"), gdal.GA_ReadOnly)
                            #     wat_1 = dataset.ReadAsArray()
                            #     dataset = None
                            #     dataset = gdal.Open(os.path.join(folder_test, str(n_+1)+"_bare_soil.tif"), gdal.GA_ReadOnly)
                            #     soil_1 = dataset.ReadAsArray()
                            #     dataset = None

                            #     dataset = gdal.Open(os.path.join(folder_test, "Patches_0.tif"), gdal.GA_ReadOnly)
                            #     pat0 = dataset.ReadAsArray()
                            #     dataset = None
                            #     dataset = gdal.Open(os.path.join(folder_test,  "Patches_1.tif"), gdal.GA_ReadOnly)
                            #     pat1 = dataset.ReadAsArray()
                            #     dataset = None
                            #     dataset = gdal.Open(os.path.join(folder_test,  "Patches_2.tif"), gdal.GA_ReadOnly)
                            #     pat2 = dataset.ReadAsArray()
                            #     dataset = None
                            #     dataset = gdal.Open(os.path.join(folder_test,  "Patches_3.tif"), gdal.GA_ReadOnly)
                            #     pat3 = dataset.ReadAsArray()
                            #     dataset = None
                            #     dataset = gdal.Open(os.path.join(folder_test,  "Patches_4.tif"), gdal.GA_ReadOnly)
                            #     pat4 = dataset.ReadAsArray()
                            #     dataset = None

                            #     pato = pat0*pat1*pat2*pat3*pat4
                            #     filtered = 1 - pato


                            #     print(x_test[n_,:,:,0].shape)
                            #     # print(np.reshape(vv_1[0,:,:]) )#, newshape = (1,s1, s2,1)).shape)
                            #     x_test[n_,:,:,0] = vv_1[0,:,:] #np.reshape()# , newshape = (1,s1, s2,1))
                            #     x_test[n_,:,:,1] = vh_1[0,:,:] #np.reshape()# , newshape = (1,s1, s2,1))
                            #     y_test[n_,:,:,0] = soil_1[0,:,:] # np.reshape( )#, newshape = (1,s1, s2,1))
                            #     y_test[n_,:,:,1] = veg_1[0,:,:] #np.reshape( )#, newshape = (1,s1, s2,1))
                            #     y_test[n_,:,:,2] = wat_1[0,:,:] #np.reshape()#, newshape = (1,s1, s2,1))


            #                 if combinations == "VV" or combinations == "VH": 
            #                     chan = 1
            #                 elif combinations == "VVaVH":
            #                     chan = 2
            #                 else: 
            #                     chan = 6
            #                 X = np.ndarray(shape=(10,size,size,chan))
            #                 J = np.ndarray(shape=(size,size,chan))
            # #                Y = np.ndarray(shape=(batch_size,size,size,1))
            #                 Y = np.ndarray(shape=(10,size,size,3))
            #                 Y1 = np.ndarray(shape=(10,size,size,2))
            #                 for k in range(0,10):
            #                     i = ecc[k]
            #                     H = np.load(os.path.join(folder_train, 'X_val_' + str(i) + '.npy'))  
                                                    
            #                     # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif')) 
            #                     if combinations == "VV":                    
            #                         X[k,:,:,:] = np.reshape(H[:,:,1], newshape=(size,size,chan))
            #                     elif combinations == "VH":                    
            #                         X[k,:,:,:] = np.reshape(H[:,:,4], newshape=(size,size,chan))
            #                     elif combinations == "VVaVH":     
            #                         J[:,:,0] = H[:,:,1]
            #                         J[:,:,1] = H[:,:,4]               
            #                         X[k,:,:,:] = np.reshape(J, newshape=(size,size,chan))
            #                         # X[k,:,:,1] = np.reshape(H[:,:,choosen_band[1]], newshape=(size,size,chan))
            #                     else:                    
            #                         X[k,:,:,:] = np.reshape(H[:,:,:], newshape=(size,size,chan))
            #                     # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.tif'))
            #                     Y_H = np.load(os.path.join(folder_train, 'Y_val_' + str(i) + '.npy'))
            #                     # print(np.max(Y_H))
            #                     # print(Y_H.shape)
            #                     Y[k,:,:,:] = np.reshape(Y_H[:,:,:3], newshape=(size,size,3))
            #                     Y1[k,:,:,:] = np.reshape(Y_H[:,:,3:], newshape=(size,size,2))
            #                     # Y[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.npy'))
            #                 x_test2 = X.astype('float32')
            #                 y_test1 = Y.astype('float32')
            #                 opt_test = Y1.astype('float32')
                            model.load_weights(date_2 + name_inputs +combinations + "_Supervised_Post_TriVV" + BACKBONE + "_" + name_model + "_model_wIoU"+ str(size) +".h5")

                            mm_p = x_test.shape[1]//128
                            nn_p = x_test.shape[2]//128
                            x_val_1 = np.ndarray(shape=(1, mm_p*128, nn_p*128, N), dtype='float32')
                            y_val_1 = np.ndarray(shape=(1, mm_p*128, nn_p*128, 3), dtype='float32')
                            pred_val_1 = np.ndarray(shape=(1, mm_p*128, nn_p*128, 3), dtype='float32')
                            pred_val_1p = np.ndarray(shape=(1, mm_p*128, nn_p*128, 3), dtype='float32')
                            if combinations == "VV" or combinations == "VH":
                                pred_val_x1 = np.ndarray(shape=( mm_p*128, nn_p*128), dtype='float32')
                            elif combinations == "Tri_one" or combinations == "VVaVH": 
                                pred_val_x1 = np.ndarray(shape=( mm_p*128, nn_p*128,2), dtype='float32')
                            else:
                                pred_val_x1 = np.ndarray(shape=( mm_p*128, nn_p*128,3), dtype='float32')
                            pred_val_x2 = np.ndarray(shape=( mm_p*128, nn_p*128), dtype='float32')
                            k = 1
                            print(x_test.shape)
                            for m_pred in range(0, mm_p):
                                for n_pred in range(0,nn_p):
                                    m_pred1 = m_pred +1 
                                    n_pred1 = n_pred +1 
                                    x_test2 = x_test[:x_test.shape[0],(m_pred1-1)*128:m_pred1*128, (n_pred1-1)*128:n_pred1*128, :num[combinations]] # x_test[:,:m_pred1*128, :n_pred1*128, :]
                                    y_test1 =  y_test[:x_test.shape[0],(m_pred1-1)*128:m_pred1*128, (n_pred1-1)*128:n_pred1*128, :3] # y_val[:,:m_pred1*128, :n_pred1*128, :3]
                                    preds_val1 = model.predict(x_test2, verbose=1) #[:x_test.shape[0],:,:, :]
                                    # print(preds_val1.shape)
                                    preds_val = preds_val1[2]
                                    preds_valx = preds_val1[1]
                                    preds_val_pre = preds_val1[0]
                                    mix_LAYERS_2 = np.squeeze(preds_val)
                                    mix_LAYERS_2 = np.argmax(mix_LAYERS_2, axis = -1)
                                    preds_val[:,:,:,0] = mix_LAYERS_2 == 0
                                    preds_val[:,:,:,1] = mix_LAYERS_2 == 1
                                    preds_val[:,:,:,2] = mix_LAYERS_2 == 2
                                    mix_LAYERS_2p = np.squeeze(preds_val_pre)
                                    mix_LAYERS_2p = np.argmax(mix_LAYERS_2p, axis = -1)
                                    preds_val_pre[:,:,:,0] = mix_LAYERS_2p == 0
                                    preds_val_pre[:,:,:,1] = mix_LAYERS_2p == 1
                                    preds_val_pre[:,:,:,2] = mix_LAYERS_2p == 2
                                    
                                    # preds_val = mix_LAYERS_2
                                    x_val_1[:,(m_pred1-1)*128:m_pred1*128, (n_pred1-1)*128:n_pred1*128,:] = x_test2[0,:,:,:]
                                    y_val_1[:,(m_pred1-1)*128:m_pred1*128, (n_pred1-1)*128:n_pred1*128,:] = y_test1[0,:,:,:]
                                    pred_val_1[:,(m_pred1-1)*128:m_pred1*128, (n_pred1-1)*128:n_pred1*128,:] = preds_val[0,:,:,:3]
                                    pred_val_1p[:,(m_pred1-1)*128:m_pred1*128, (n_pred1-1)*128:n_pred1*128,:] = preds_val_pre[0,:,:,:3]

                                    # pred_val_11 = preds_val[k,:,:,2:]
                                    if combinations == "VV" or combinations == "VH":
                                        pred_val_x1[(m_pred1-1)*128:m_pred1*128, (n_pred1-1)*128:n_pred1*128] = preds_valx[0,:,:,0]
                                    elif combinations == "Tri_one" or combinations == "VVaVH":  
                                        pred_val_x1[(m_pred1-1)*128:m_pred1*128, (n_pred1-1)*128:n_pred1*128,:] = preds_valx[0,:,:,:2]
                                    else:
                                        pred_val_x1[(m_pred1-1)*128:m_pred1*128, (n_pred1-1)*128:n_pred1*128,:] = preds_valx[0,:,:,:3]
                                        # pred_val_x2[(m_pred1-1)*128:m_pred1*128, (n_pred1-1)*128:n_pred1*128] = preds_valx[0,:,:,1]
                                    
                            path_results = r"C:\Users\massi\Downloads\segmentation_models-master\images\complete_unsupervised_results_2020_20\\"
                            if not os.path.exists(path_results):
                                os.makedirs(path_results)
                            name_inputs1 = combinations + "_Testing1_" + name_inputs + "_" + date_3
                            print(pred_val_1.shape)
                            print(x_val_1.shape)
                            ndvi = y_val_1.astype('float32')
                            im = path_results + name_inputs1 + combinations + "_" + BACKBONE+ date_3  + '_' + name_model + '_target_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                            imsave(im, ndvi)
                            
                            ndvi2 = pred_val_1.astype('float32')
                            im2 =  path_results + name_inputs1 + combinations + "_" + BACKBONE +date_3  + '_' + name_model + '_output2_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                            imsave(im2, ndvi2)

                            ndvi3 = pred_val_1p.astype('float32')
                            im3 =  path_results + name_inputs1 + combinations + "_" + BACKBONE +date_3  + '_' + name_model + '_output1_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                            imsave(im3, ndvi3)


                            if combinations == "VV":
                                ndvi3 = pred_val_x1.astype('float32')
                                im3 =  path_results + name_inputs1 + combinations + "_" + BACKBONE +date_3  + '_' + name_model + '_est_inputVV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                                imsave(im3, ndvi3)

                                ndvi4 = x_val_1[:,:,:,0].astype('float32')
                                im4 = path_results + name_inputs1 + combinations + "_" + BACKBONE+ date_3  + '_' + name_model + '_inputVV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                                imsave(im4, ndvi4)
                            else: 
                                ndvi3 = pred_val_x1.astype('float32')
                                im3 =  path_results + name_inputs1 + combinations + "_" + BACKBONE +date_3  + '_' + name_model + '_est_inputVV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                                imsave(im3, ndvi3)

                                # ndvi4 = x_val_1[:,:,:,0].astype('float32')
                                # im4 = path_results + name_inputs1 + combinations + "_" + BACKBONE+ date_3  + '_' + name_model + '_inputVV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                                # imsave(im4, ndvi4)

                                # ndvi5 = pred_val_x2.astype('float32')
                                # im5 =  path_results + name_inputs1 + combinations + "_" + BACKBONE +date_3  + '_' + name_model + '_est_inputVH_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                                # imsave(im5, ndvi5)

                                # ndvi6 = x_val_1[:,:,:,1].astype('float32')
                                # im6 = path_results + name_inputs1 + combinations + "_" + BACKBONE+ date_3  + '_' + name_model + '_inputVH_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                                # imsave(im6, ndvi6)


                            #########TEST###################à///////////// TESTING
                            # m_pred1 = 2 # x_test.shape[1]//128
                            # n_pred1 = 1 # x_test.shape[2]//128
                            # N_samples = x_test.shape[0]# 8 # len(x_val)
                            # preds_val = np.ndarray(shape=(N_samples, m_pred1*128, n_pred1*128, Out_classes), dtype='float32')
                            
                            # if combinations == "VV":
                            #     x_test1 = np.ndarray(shape=(N_samples, m_pred1*128, n_pred1*128, 1), dtype='float32')
                            #     x_test1 = x_test[:,(m_pred1-1)*128:m_pred1*128, :n_pred1*128, 1] # x_test[:,:m_pred1*128, :n_pred1*128, 1]
                            #     x_test2 = np.reshape(x_test1, newshape = (N_samples,  m_pred1*128, n_pred1*128, 1))
                            # elif combinations == "VH":
                            #     x_test1 = np.ndarray(shape=(N_samples, m_pred1*128, n_pred1*128, 1), dtype='float32')
                            #     x_test1 = x_test[:,(m_pred1-1)*128:m_pred1*128, :n_pred1*128, 4] # x_test[:,:m_pred1*128, :n_pred1*128, 4]
                            #     x_test2 = np.reshape(x_test1, newshape = (N_samples,  m_pred1*128, n_pred1*128, 1))
                            # elif combinations == "VVaVH":
                            #     x_test2 = np.ndarray(shape=(N_samples, m_pred1*128, n_pred1*128, 2), dtype='float32')

                            #     x_test2[:,:,:,0] = x_test[:,(m_pred1-1)*128:m_pred1*128, :n_pred1*128, 1] #x_test[:,:m_pred1*128, :n_pred1*128, 1]
                            #     x_test2[:,:,:,1] = x_test[:,(m_pred1-1)*128:m_pred1*128, :n_pred1*128, 4] # x_test[:,:m_pred1*128, :n_pred1*128, 4]
                            # elif combinations == "Tri":
                            #     x_test2 = x_test[:x_test.shape[0],(m_pred1-1)*128:m_pred1*128, :n_pred1*128, :] # x_test[:,:m_pred1*128, :n_pred1*128, :]
                            # y_test1 =  y_test[:x_test.shape[0],(m_pred1-1)*128:m_pred1*128, :n_pred1*128, :3] # y_val[:,:m_pred1*128, :n_pred1*128, :3] #

                            # print(np.max(x_test))
                            # print(np.min(x_test))

                            # # opt_test = y_test[:,:m_pred1*128, :n_pred1*128, 3:]
                            # # # x_test2 = x_val
                            # # for k_pred in range(0,x_test.shape[1]-127,128):
                            # #     for k2_pred in range(0,x_test.shape[2]-127,128):

                            # #         preds_val[:,k_pred:k_pred + 128, k2_pred:k2_pred + 128, :] = model.predict(x_test2[:,k_pred:k_pred + 128, k2_pred:k2_pred + 128, :], verbose=1)
                            # preds_val1 = model.predict(x_test2[:x_test.shape[0],:,:, :], verbose=1)
                            # preds_val = preds_val1[0]
                            # preds_valx = preds_val1[1]
                            # mix_LAYERS_2 = np.squeeze(preds_val)
                            # mix_LAYERS_2 = np.argmax(mix_LAYERS_2, axis = -1)
                            
                            # preds_val[:,:,:,0] = mix_LAYERS_2 == 0
                            # preds_val[:,:,:,1] = mix_LAYERS_2 == 1
                            # preds_val[:,:,:,2] = mix_LAYERS_2 == 2
                            # # preds_val[:,:,:,3] = mix_LAYERS_2 == 3
                            # # preds_val[:,:,:,4] = mix_LAYERS_2 == 4
                            # path_results = r"C:\Users\massi\Downloads\segmentation_models-master\images\unsupervised_results_2020\\"
                            # if not os.path.exists(path_results):
                            #     os.makedirs(path_results)
                            # name_inputs1 = "1Test_" + name_inputs
                            # for k in range(0,x_test.shape[0]): #x_test2.shape[0],x_test2.shape[0]//10): #x_test2.shape[0]):#x_test.shape[0],int(x_test.shape[0]/100)): 
                            #     x_val_1 = x_test2[k,:,:,:]
                            #     y_val_1 = y_test1[k,:,:,:]
                            #     pred_val_1 = preds_val[k,:,:,:3]
                            #     # pred_val_11 = preds_val[k,:,:,2:]
                            #     pred_val_x = preds_valx[k,:,:,:]
                            #     # opt_test_1 = opt_test[k,:,:,0]
                            #     # opt_test_2 = opt_test[k,:,:,1]
                            #     # print("NDVI")
                            #     # print(np.max(pred_val_1))
                            #     # print(np.max(y_val_1))
                            #     # print(y_val_1.shape)
                            #     # print(x_val_1.shape)
                            #     print(pred_val_1.shape)
                            #     ndvi = y_val_1.astype('float32')
                            #     im = path_results + name_inputs1 + combinations + "_" + BACKBONE+ date_3  + '_' + name_model + '_target_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                                
                            #     imsave(im, ndvi)
                                
                            #     ndvi2 = pred_val_1.astype('float32')
                            #     im2 =  path_results + name_inputs1 + combinations + "_" + BACKBONE +date_3  + '_' + name_model + '_output_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                            #     imsave(im2, ndvi2)

                            #     # ndvi2 = pred_val_11.astype('float32')
                            #     # im2 =  path_results + name_inputs1 + combinations + "_" + BACKBONE +date_3  + '_' + name_model + '_output2_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                            #     # imsave(im2, ndvi2)


                            #     ndvi2 = pred_val_x.astype('float32')
                            #     im2 =  path_results + name_inputs1 + combinations + "_" + BACKBONE +date_3  + '_' + name_model + '_est_input_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                            #     imsave(im2, ndvi2)


                            #     # y_val_1 = y_test1[k,:,:,1]
                            #     # pred_val_1 = preds_val[k,:,:,1]
                            #     # # print("MNDWI")
                            #     # # print(np.max(pred_val_1))
                            #     # # print(np.max(y_val_1))
                            #     # # print(y_val_1.shape)
                            #     # # print(pred_val_1.shape)
                            #     # ndvi = y_val_1.astype('float32')
                            #     # im = combinations + "_" + BACKBONE+ date_3  + '_' + name_model + '_targetm_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                                
                            #     # imsave(im, ndvi)
                                
                            #     # ndvi2 = pred_val_1.astype('float32')
                            #     # im2 =  name_inputs + combinations + "_" + BACKBONE +date_3  + '_' + name_model + '_outputm2_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                            #     # imsave(im2, ndvi2)
                            #     if name_model.find("shallow") != -1: 
                            #         ndvi3 = opt_test_1[:,:].astype('float32')
                            #         im3 = path_results + combinations + "_" + BACKBONE + date_3  +'_' + name_model + '_NDVI_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                            #         imsave(im3, ndvi3)
                            #         ndvi3 = opt_test_2[:,:].astype('float32')
                            #         im3 = path_results + combinations + "_" + BACKBONE + date_3  +'_' + name_model + '_MNDWI_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                            #         imsave(im3, ndvi3)
                            #         if N == 1: 
                            #             ndvi3 = x_val_1[:,:,0].astype('float32')
                            #             im3 = path_results + combinations + "_" + BACKBONE + date_3  +'_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                            #             imsave(im3, ndvi3)
                            #         elif N == 2:
                            #             ndvi3 = x_val_1[:,:,0].astype('float32')
                            #             im3 = path_results + combinations + "_" + BACKBONE + date_3  +'_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                            #             imsave(im3, ndvi3)
                            #             ndvi4 = x_val_1[:,:,1].astype('float32')
                            #             im4 =  combinations + "_" + BACKBONE +date_3  + '_' + name_model + '_VH_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                            #             imsave(im4, ndvi4)
                            #         else: 
                            #             ndvi3 = x_val_1[:,:,1].astype('float32')
                            #             im3 = path_results + combinations + "_" + BACKBONE + date_3  +'_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                            #             imsave(im3, ndvi3)
                            #             ndvi4 = x_val_1[:,:,4].astype('float32')
                            #             im4 =  path_results + combinations + "_" + BACKBONE +date_3  + '_' + name_model + '_VH_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                            #             imsave(im4, ndvi4)
