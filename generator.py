# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:15:46 2019

@author: franc
"""

import numpy as np
import random as rd
import os
from scipy.io import loadmat
from keras.utils import np_utils


def train_generator(batch_size,num_classes):
    n = 256
    num_train_sample = len(os.listdir('./Dataset_1/Input/Train/'))
    index = np.array(range(num_train_sample))
    batch_features = np.zeros((batch_size,n,n,1))
    batch_labels_ = np.zeros((batch_size,n,n,num_classes))
    path_ref = './Dataset_1/Reference/Train/'
    path_inp = './Dataset_1/Input/Train/'
    elem = os.listdir(path_ref)

    while True:
            rd.shuffle(index)
            for i in range(batch_size):
                ref_ = loadmat(path_ref + elem[index[i]])
                ref = ref_['wrap_count']
                ref = ref + 48
                inp_ = loadmat(path_inp + elem[index[i]])
                inp = inp_['wrapp']
                inp = np.reshape(inp,(inp.shape[0],inp.shape[1],1))                
                ref = np.reshape(ref,(inp.shape[0],inp.shape[1],1))
                batch_features[i] = inp
                batch_labels_[i] = ref
                
                
                batch_labels_[i] = np_utils.to_categorical(ref,num_classes)

            yield batch_features,batch_labels_
            
            
def val_generator(batch_size,num_classes):
    n = 256
    num_val_sample = len(os.listdir('./Dataset_1/Input/Val/'))
    index = np.array(range(num_val_sample))
    batch_features = np.zeros((batch_size,n,n,1))
    batch_labels_ = np.zeros((batch_size,n,n,num_classes))
    path_ref = './Dataset_1/Reference/Val/'
    path_inp = './Dataset_1/Input/Val/'
    elem = os.listdir(path_ref)
    while True:
            rd.shuffle(index)
            for i in range(batch_size):
                ref_ = loadmat(path_ref + elem[index[i]])
                ref = ref_['wrap_count']
                ref = ref + 48
                inp_ = loadmat(path_inp + elem[index[i]])
                inp = inp_['wrapp']
                inp = np.reshape(inp,(inp.shape[0],inp.shape[1],1))                
                ref = np.reshape(ref,(inp.shape[0],inp.shape[1],1))
                batch_features[i] = inp
                batch_labels_[i] = ref
                
                batch_labels_[i] = np_utils.to_categorical(ref,num_classes)
                
            yield batch_features,batch_labels_ 
            
def test_generator(batch_size,num_classes):
    num_val_sample = len(os.listdir('./Dataset/Input/Test/'))
    index = np.array(range(num_val_sample))
    batch_features = np.zeros((batch_size,128,128,1))
    batch_labels_ = np.zeros((batch_size,128,128,num_classes))
    path_ref = './Dataset/Reference/Test/'
    path_inp = './Dataset/Input/Test/'
    elem = os.listdir(path_ref)
    while True:
            rd.shuffle(index)
            for i in range(batch_size):
                ref_ = np.load(path_ref + elem[index[i]])
                ref = ref_
                ref = ref + 19
                inp_ = np.load(path_inp + elem[index[i]])
                inp = inp_
                inp = np.reshape(inp,(inp.shape[0],inp.shape[1],1))                
                ref = np.reshape(ref,(inp.shape[0],inp.shape[1],1))
                batch_features[i] = inp
                batch_labels_[i] = ref
                
                batch_labels_[i] = np_utils.to_categorical(ref,num_classes)
                
            yield batch_features,batch_labels_ 