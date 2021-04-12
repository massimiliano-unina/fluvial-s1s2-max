# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 17:41:02 2019

@author: massi
"""
import tensorflow as tf 
import numpy as np 
from tensorflow import keras
from tensorflow.contrib import lite

model = keras.Sequential([keras.layers.Dense(units= 1, input_shape = [1])])
model.compile(optimizer='sgd',loss='mean_squared_error')

xs = np.array([1.0, 5.0, 7.0, 2.0, 6.0], dtype=float)
ys = np.array([1.0, 9.0, 13.0, 3.0, 11.0], dtype=float)

model.fit(xs,ys, epochs=100)

print(model.predict([10.0]))

keras_file = "linear.h5"
keras.models.save_model(model,keras_file)

converter = lite.TocoConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("linear.tflite","wb").write(tflite_model)

