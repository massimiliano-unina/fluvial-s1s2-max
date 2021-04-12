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
folder_1 = r"C:\Users\massi\Downloads\sentinel1_Salerno\Albufera-20200415T192751Z-001\Albufera\File_Training\Input\MNDWI_NDWI\\" #r"D:\Albufera\\"#

## testing a parte i
folder_test = folder_1 #r"D:\Works\Albufera-SemanticSegmentation\Testing\S2_S1\\"
size = 128
size_t = 144 # 128 #
n_epochs = 5 # 1 #1# 20
n_batch = 32
LR_MAX = 0.02 # 0.005
comb = ["VVaVH"] #["VV", "VH","VVaVH","Tri"] # ["VV", "VH","VVaVH", "Tri"] #  ["VV", "VH", , "VVaVHaSum","VVaVHaRatio", "VVaVHaDiff","Total"]#"Ratio",  #["VVaVH"]#Ratio"]# ["VVaVH"]##
# num = [1,1,2]#,3,3,3,5]#1,#[2]# 
num = {"VH": 1, "VV": 1, "VVaVH":2, "Tri": 6}
#class_weight = {0: 50., 1: 1., 2: 1.}
# class_weight = [40.0, 1.0, 1.0]

# x_traina, y_traina, x_vala, y_vala,x_train2a, y_train2a, x_val2a, y_val2a = load_data(folder_1, size,size_t,indo,indo1)
# print(x_traina.shape)
# print(x_vala.shape)
# np.savez("train_MetriAgriFOR.npz",x_traina = x_traina, y_traina = y_traina, x_vala = x_vala, y_vala = y_vala,x_train2a = x_train2a, y_train2a = y_train2a, x_val2a = x_val2a, y_val2a = y_val2a) #,  x_gtrain = x_gtrai y_train = y_trai  x_gval = x_gval, y_val = y_val)    

# train_MetriAgriFOR = np.load("train_MetriAgriFOR.npz")
dic = {}
for date_alb in range(1): 
    Out_classes = 3
    date_2 = str(date_alb + 1)
    # train_MetriAgriFOR = np.load("train_MetriAgriFOR_"+ date_2 + "Date.npz")

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
    files_test = ["test_intradate_beta.npz"]#, "test_intradate_gamma.npz", "test_intradate_beta.npz"]
    dic_name_inputs = {"test_intradate_beta.npz":"sigma_", "test_intradate_gamma.npz":"gamma_", "test_intradate_a.npz":"beta_"}
    folder_trains = {"test_intradate_beta.npz":r"C:\Users\massi\Downloads\sentinel1_Salerno\Albufera-20200415T192751Z-001\Albufera\File_Training\Input\MNDWI_NDWI\\", "test_intradate_gamma.npz":r"D:\Albufera_2019_processed\Gamma2\\", "test_intradate_a.npz":r"D:\Albufera_2019_processed\Beta2\\"}

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
        # test_MetriAgriFOR_with_summer_augmentation = np.load(file_test)
        name_inputs =  dic_name_inputs[file_test]
        # x_test = test_MetriAgriFOR_with_summer_augmentation["x_test"] 
        # y_test = test_MetriAgriFOR_with_summer_augmentation["y_test"]#, x_gtrain = x_gtrain, y_train = y_train,  x_gval = x_gval, y_val = y_val)    

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
                BACKBONE = backs[k_back]# 
                preprocess_input = get_preprocessing(BACKBONE)        ## define model and chose between following models: 
                
                networks =["DenseUNet"]#["shallow_CNN"]# ["Fractal_Net", "FractalNet","shallow_CNN" , "Unet","Linknet","SegNet","FPN"] # ["SegNet","FPN"]#   "NestNet" "shallow_CNN"] #  ["SVM", "RF", "GBC", 
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
                    
                        # outputs = Convolution2D(3, 1, 1, activation=activate3, border_mode='same')(conv9b)
                        outputs = Conv2D(Out_classes, kernel_size=1, activation=activate3, padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv9b)
                        model = Model(inputs=inputs, outputs=outputs)      
                    elif k_mod == "DenseUNet":           
                        f = 8
                        # N = x_train.shape[-1]
                        inputs = Input((None, None, N))
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
                        outputs = Conv2D(Out_classes, kernel_size=1, activation='softmax', padding= 'same', kernel_initializer = keras.initializers.glorot_normal(seed=None))(conv9)
                        model = Model(inputs=inputs, outputs=outputs)    
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
            #         if k_mod == "SVM" or k_mod == "RF" or k_mod == "GBC" :
            #             model.fit(X, y)
            #         else: 
            #             # if k_mod == "shallow_CNN":  
            #             #     model.load_weights( name_inputs +combinations + "_" + BACKBONE + "_" + name_model_pre + "_model_wIoU"+ str(size) +".h5")
            #             Adamax = Adam(lr = LR_MAX, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            #             ##### lr=0.0001
            #             model.compile(loss= bce_jaccard_loss , metrics = [iou_score], optimizer = Adamax) # 
            #             # dice_loss categorical_crossentropy  bce_jaccard_loss  cce_jaccard_loss cce_dice_loss , metrics=[iou_score], optimizer=Adamax)

            #             # model.compile(loss=MAE, metrics = [MAE], optimizer = Adamax) # keras.losses.sparse_c
            # #            model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
            # #            
            #             callbacks = [
            #                 EarlyStopping(patience=10, verbose=1),
            #                 ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
            #                 ModelCheckpoint( name_inputs + combinations + "_" + BACKBONE + "_" + name_model + "_temporary_model_IoU"+ str(size) +".h5", verbose=1, save_best_only=True, save_weights_only=True),
            #                 tensorboard
            #                 # time_callback
            #             ]

            #             print("number_train" + str(number_train))
            #             s_p_e = number_train//n_batch # x_train.shape[0]//n_batch # 
            #             train_val_p2 = n_batch*s_p_e #x_train.shape[0] # n_batch*50#250# #n_batch*500 
            #             val_p2 = n_batch*10
            #             val_pe = val_p2//n_batch
            #             print(type(s_p_e))

            #             model.fit_generator(train_generator_2(train_val_p2, number_train, n_batch,folder_train, size,N, combinations), validation_data = val_generator_2(train_val_p2, number_val, n_batch,folder_train, size,N, combinations), validation_steps = val_pe, steps_per_epoch = s_p_e, epochs = n_epochs,  callbacks = callbacks, class_weight=[0.4,0.3, 0.3])

            #             # model.fit_generator(train_generator_3(train_val_p2, number_train, n_batch,folder_train, size,N, combinations, x_train, y_train), validation_data = val_generator_3(train_val_p2, number_val, n_batch,folder_train, size,N, combinations, x_train, y_train), validation_steps = val_pe, steps_per_epoch = s_p_e, epochs = n_epochs,callbacks = callbacks) #class_weight=[0.4,0.5, 0.1], callbacks = callbacks)

            #             ## fit model
            #             # if name_model == "PSPNet":
            #             #     model.fit(
            #             #         x=x_train2,
            #             #         y=y_train2,
            #             #         batch_size=n_batch,
            #             #         epochs=n_epochs,
            #             #         class_weight = class_weight,
            #             #         validation_data=(x_val2, y_val2),callbacks = callbacks
            #             #     )
            #             # else: 
            #             #     model.fit(
            #             #         x=x_train,
            #             #         y=y_train,
            #             #         batch_size=n_batch,
            #             #         epochs=n_epochs,
            #             #         class_weight = class_weight,
            #             #         validation_data=(x_val, y_val),callbacks = callbacks
            #             #     )
            #             times = time.time() - timer 
            #             dic_times = {}
            #             dic_times['times'] = times
            #             savemat( name_inputs +combinations + "_" + BACKBONE + '_' + name_model + '_times_MNDWI.mat', dic_times)
            #             model.save_weights( name_inputs +combinations + "_MNDWI_" + BACKBONE + "_" + name_model + "_model_wIoU"+ str(size) +".h5")
        ############END TRAINING#############           
                    
                    # Load best model
                    if k_mod == "SVM" or k_mod == "RF" or k_mod == "GBC": 
                        pkl.dump(model, open(output_model, "wb"))
                        pred = model.predict(X[:,10])
                        precision = metrics.precision_score(y[:,10], pred, average='weighted', labels=np.unique(pred))
                        recall = metrics.recall_score(y[:,10], pred, average='weighted', labels=np.unique(pred))
                        f1 = metrics.f1_score(y[:,10], pred, average='weighted', labels=np.unique(pred))
                        accuracy = metrics.accuracy_score(y[:,10], pred)
                    else: 
                        # model.load_weights( name_inputs +combinations + "_" + BACKBONE + "_" + name_model + "_temporary_model_IoU"+ str(size) +".h5")
                        model.load_weights( name_inputs +combinations + "_MNDWI_" + BACKBONE + "_" + name_model + "_model_wIoU"+ str(size) +".h5")
                    #    model.evaluate(x_val, y_val, verbose=1)
                        # Predict on train, val and test

            #             if name_model == "PSPNet":
            #                 preds_train = model.predict(x_train2, verbose=1)
            #                 preds_val = model.predict(x_val2, verbose=1)
            #                 for k in range(0,x_val.shape[0],int(x_val.shape[0]/100)):
            #                     x_val_1 = x_val2[k,:,:,:]
            #                     y_val_1 = y_val2[k,:,:,:]
            #                     pred_val_1 = preds_val[k,:,:,:]
                                
            #                     ndvi = y_val_1.astype('float32')
            #                     im = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_target_'+ str(k) + '_wpatches'+ str(size) +'.tif'
            #                     imsave(im, ndvi)
                                
            #                     ndvi2 = pred_val_1.astype('float32')
            #                     im2 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_output_'+ str(k) + '_wpatches'+ str(size) +'.tif'
            #                     imsave(im2, ndvi2)
            #                     if N == 1:
            #                         ndvi3 = x_val_1[:,:,0].astype('float32')
            #                         im3 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
            #                         imsave(im3, ndvi3)
            #                     else: 
            #                         ndvi3 = x_val_1[:,:,0].astype('float32')
            #                         im3 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
            #                         imsave(im3, ndvi3)
            #                         ndvi4 = x_val_1[:,:,1].astype('float32')
            #                         im4 =  comb[combinations] + "_" + BACKBONE + '_' + name_model + '_VH_'+ str(k) + '_wpatches'+ str(size) +'.tif'
            #                         imsave(im4, ndvi4)
            #             else:
            # #                preds_train = model.predict(x_train, verbose=1)
            #                 preds_val = model.predict(x_test, verbose=1)
                            
            #                 for k in range(0,x_val.shape[0],int(x_val.shape[0]/100)): 
            #                     x_val_1 = x_test[k,:,:,:]
            #                     y_val_1 = y_test[k,:,:,:]
            #                     pred_val_1 = preds_val[k,:,:,:]
                                
            #                     ndvi = y_val_1.astype('float32')
            #                     im = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_target_'+ str(k) + '_wpatches'+ str(size) +'.tif'
            #                     imsave(im, ndvi)
                                
            #                     ndvi2 = pred_val_1.astype('float32')
            #                     im2 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_output_'+ str(k) + '_wpatches'+ str(size) +'.tif'
            #                     imsave(im2, ndvi2)
            #                     if N == 1: 
            #                         ndvi3 = x_val_1[:,:,0].astype('float32')
            #                         im3 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
            #                         imsave(im3, ndvi3)
            #                     else: 
            #                         ndvi3 = x_val_1[:,:,0].astype('float32')
            #                         im3 = comb[combinations] + "_" + BACKBONE + '_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
            #                         imsave(im3, ndvi3)
            #                         ndvi4 = x_val_1[:,:,1].astype('float32')
            #                         im4 =  comb[combinations] + "_" + BACKBONE + '_' + name_model + '_VH_'+ str(k) + '_wpatches'+ str(size) +'.tif'
            #                         imsave(im4, ndvi4)


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
                            import imageio
                            path_test = r"C:\Users\massi\Downloads\sentinel1_Salerno\Albufera-20200415T192751Z-001\Albufera\File_Training\Input\VV_VH\\"
                            data_n = ["26022020.tif", "02022020.tif", "05022018.tif", "06022018.tif", "07022020.tif", "08022020.tif", "12022018.tif", "14022020.tif", "17022018.tif", "18022018.tif", "19022020.tif", "20022020.tif", "24022018.tif"]
                            if combinations == "VV" or combinations == "VH": 
                                chan = 1
                            elif combinations == "VVaVH":
                                chan = 2
                            else: 
                                chan = 6
                            for k in range(0,len(data_n)):
                                data = data_n[k]
                                H = imageio.imread(os.path.join(path_test, "VV_" + data))
                                [size1, size2] = H.shape
                                X = np.ndarray(shape=(1,size1,size2,chan))
                                J = np.ndarray(shape=(size1,size2,chan))
                #                Y = np.ndarray(shape=(batch_size,size,size,1))
                                Y = np.ndarray(shape=(1,size1,size2,2))
                                Y1 = np.ndarray(shape=(1,size1,size2,2))
                                # H = np.load(os.path.join(folder_train, 'X_val_' + str(i) + '.npy'))  
                                                    
                                # H = imageio.imread(os.path.join(data_folder, 'X_train_' + str(ind[i + k]) + '.tif')) 
                                if combinations == "VV":                    
                                    X[k,:,:,:] = np.reshape(H[:,:,1], newshape=(size,size,chan))
                                elif combinations == "VH":                    
                                    X[k,:,:,:] = np.reshape(H[:,:,4], newshape=(size,size,chan))
                                elif combinations == "VVaVH":     
                                    J[:,:,0] = imageio.imread(os.path.join(path_test, "VV_" + data))
                                    J[:,:,1] = imageio.imread(os.path.join(path_test, "VH_" + data))            
                                    X[0,:,:,:] = np.reshape(J, newshape=(size1,size2,chan))
                                    # X[k,:,:,1] = np.reshape(H[:,:,choosen_band[1]], newshape=(size,size,chan))
                                else:                    
                                    X[k,:,:,:] = np.reshape(H[:,:,:], newshape=(size,size,chan))
                                # Y[k,:,:,:] = imageio.imread(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.tif'))
                                # Y_H = np.load(os.path.join(folder_train, 'Y_val_' + str(i) + '.npy'))
                                # # print(np.max(Y_H))
                                # # print(Y_H.shape)
                                # Y[k,:,:,:] = np.reshape(Y_H[:,:,:], newshape=(size,size,2))
                                # Y1[k,:,:,:] = np.reshape(Y_H[:,:,:], newshape=(size,size,2))
                                # # Y[k,:,:,:] = np.load(os.path.join(data_folder, 'Y_train_' + str(ind[i + k]//10) + '.npy'))
                                x_test2 = X.astype('float32')
                                # y_test1 = Y.astype('float32')
                                # opt_test = Y1.astype('float32')

                                m_pred1 = x_test2.shape[1]//128
                                n_pred1 = x_test2.shape[2]//128
                                # N_samples = 10 # len(x_val)
                                preds_val = np.ndarray(shape=(1, m_pred1*128, n_pred1*128, Out_classes), dtype='float32')
                            
                            # if combinations == "VV":
                            #     x_test1 = np.ndarray(shape=(N_samples, m_pred1*128, n_pred1*128, 1), dtype='float32')
                            #     x_test1 = x_test[:,:m_pred1*128, :n_pred1*128, 1] # x_test[:,:m_pred1*128, :n_pred1*128, 1]
                            #     x_test2 = np.reshape(x_test1, newshape = (N_samples,  m_pred1*128, n_pred1*128, 1))
                            # elif combinations == "VH":
                            #     x_test1 = np.ndarray(shape=(N_samples, m_pred1*128, n_pred1*128, 1), dtype='float32')
                            #     x_test1 = x_test[:,:m_pred1*128, :n_pred1*128, 4] # x_test[:,:m_pred1*128, :n_pred1*128, 4]
                            #     x_test2 = np.reshape(x_test1, newshape = (N_samples,  m_pred1*128, n_pred1*128, 1))
                            # elif combinations == "VVaVH":
                            #     x_test2 = np.ndarray(shape=(N_samples, m_pred1*128, n_pred1*128, 2), dtype='float32')

                            #     x_test2[:,:,:,0] = x_test[:,:m_pred1*128, :n_pred1*128, 1] #x_test[:,:m_pred1*128, :n_pred1*128, 1]
                            #     x_test2[:,:,:,1] = x_test[:,:m_pred1*128, :n_pred1*128, 4] # x_test[:,:m_pred1*128, :n_pred1*128, 4]
                            # elif combinations == "Tri":
                            #     x_test2 = x_test[:,:m_pred1*128, :n_pred1*128, :] # x_test[:,:m_pred1*128, :n_pred1*128, :]
                            # y_test1 =  y_test[:,:m_pred1*128, :n_pred1*128, :3] # y_val[:,:m_pred1*128, :n_pred1*128, :3] #
                            # opt_test = y_test[:,:m_pred1*128, :n_pred1*128, 3:]
                            # # # x_test2 = x_val
                                for k_pred in range(0,x_test2.shape[1]-127,128):
                                    for k2_pred in range(0,x_test2.shape[2]-127,128):

                                        preds_val[:,k_pred:k_pred + 128, k2_pred:k2_pred + 128, :] = model.predict(x_test2[:,k_pred:k_pred + 128, k2_pred:k2_pred + 128, :], verbose=1)
                            # preds_val = model.predict(x_test2[:,:,:, :], verbose=1)
                                mix_LAYERS_2 = np.squeeze(preds_val)
                                mix_LAYERS_2 = np.argmax(mix_LAYERS_2, axis = -1)
                                
                                preds_val[:,:,:,0] = mix_LAYERS_2 == 0
                                preds_val[:,:,:,1] = mix_LAYERS_2 == 1
                            # preds_val[:,:,:,2] = mix_LAYERS_2 == 2


                                name_inputs1 = "Test_" + name_inputs
                                for k in range(0,1): #x_test2.shape[0],x_test2.shape[0]//10): #x_test2.shape[0]):#x_test.shape[0],int(x_test.shape[0]/100)): 
                                    x_val_1 = x_test2[k,:,:,:]
                                    # y_val_1 = y_test1[k,:,:,:]
                                    pred_val_1 = preds_val[k,:,:,:]
                                    # opt_test_1 = opt_test[k,:,:,0]
                                    # opt_test_2 = opt_test[k,:,:,1]
                                    # print("NDVI")
                                    # print(np.max(pred_val_1))
                                    # print(np.max(y_val_1))
                                    # print(y_val_1.shape)
                                    # print(x_val_1.shape)
                                    print(pred_val_1.shape)
                                    # ndvi = y_val_1[:,:,0].astype('float32')
                                    # im = name_inputs1 + combinations + "_" + BACKBONE+ date_2  + '_' + name_model + '_target_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                                    
                                    # imsave(im, ndvi)
                                    ndvi1 = pred_val_1[:,:,1].astype('float32')
                                    ndvi2 = pred_val_1[:,:,0].astype('float32')
                                    im2 = path_test +  'output_from_MNDWI_'+ data
                                    imsave(im2, ndvi1 + ndvi2)

                                    # y_val_1 = y_test1[k,:,:,1]
                                    # pred_val_1 = preds_val[k,:,:,1]
                                    # # print("MNDWI")
                                    # # print(np.max(pred_val_1))
                                    # # print(np.max(y_val_1))
                                    # # print(y_val_1.shape)
                                    # # print(pred_val_1.shape)
                                    # ndvi = y_val_1.astype('float32')
                                    # im = combinations + "_" + BACKBONE+ date_2  + '_' + name_model + '_targetm_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                                    
                                    # imsave(im, ndvi)
                                    
                                    # ndvi2 = pred_val_1.astype('float32')
                                    # im2 =  name_inputs + combinations + "_" + BACKBONE +date_2  + '_' + name_model + '_outputm2_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                                    # imsave(im2, ndvi2)
                                    if name_model.find("DenseUNet") != -1: 
                                        # ndvi3 = opt_test_1[:,:].astype('float32')
                                        # im3 = combinations + "_" + BACKBONE + date_2  +'_' + name_model + '_NDVI_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                                        # imsave(im3, ndvi3)
                                        # ndvi3 = opt_test_2[:,:].astype('float32')
                                        # im3 = combinations + "_" + BACKBONE + date_2  +'_' + name_model + '_MNDWI_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                                        # imsave(im3, ndvi3)
                                        if N == 1: 
                                            ndvi3 = x_val_1[:,:,0].astype('float32')
                                            im3 = combinations + "_" + BACKBONE + date_2  +'_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                                            imsave(im3, ndvi3)
                                        elif N == 2:
                                            ndvi3 = x_val_1[:,:,0].astype('float32')
                                            im3 = path_test +  'input_VV_' + data
                                            imsave(im3, ndvi3)
                                            ndvi4 = x_val_1[:,:,1].astype('float32')
                                            im4 =  path_test + 'input_VH_'+ data 
                                            imsave(im4, ndvi4)
                                        else: 
                                            ndvi3 = x_val_1[:,:,1].astype('float32')
                                            im3 = combinations + "_" + BACKBONE + date_2  +'_' + name_model + '_VV_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                                            imsave(im3, ndvi3)
                                            ndvi4 = x_val_1[:,:,4].astype('float32')
                                            im4 =  combinations + "_" + BACKBONE +date_2  + '_' + name_model + '_VH_'+ str(k) + '_wpatches'+ str(size) +'.tif'
                                            imsave(im4, ndvi4)
