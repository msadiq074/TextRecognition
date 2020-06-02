# -*- coding: utf-8 -*-
"""
Created on Sat May 23 04:42:43 2020

@author: msadi
"""


import os
import fnmatch
import cv2
import numpy as np
import string
import time

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.python.client import device_lib

# Check all available devices if GPU is available
print(device_lib.list_local_devices())
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

char_list = string.ascii_letters+string.digits
 
def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
        
    return dig_lst

class TextDetector:
    def __init__(self):
        self.xyz=None
        self.char_list = string.ascii_letters+string.digits
        self.inputs = Input(shape=(32,128,1))
        # convolution layer with kernel size (3,3)
        self.conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(self.inputs)
        # poolig layer with kernel size (2,2)
        self.pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(self.conv_1)
         
        self.conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(self.pool_1)
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(self.conv_2)
         
        self.conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(self.pool_2)
         
        self.conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(self.conv_3)
        # poolig layer with kernel size (2,1)
        self.pool_4 = MaxPool2D(pool_size=(2, 1))(self.conv_4)
         
        self.conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(self.pool_4)
        # Batch normalization layer
        self.batch_norm_5 = BatchNormalization()(self.conv_5)
         
        self.conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(self.batch_norm_5)
        self.batch_norm_6 = BatchNormalization()(self.conv_6)
        self.pool_6 = MaxPool2D(pool_size=(2, 1))(self.batch_norm_6)
         
        self.conv_7 = Conv2D(512, (2,2), activation = 'relu')(self.pool_6)
         
        self.squeezed = Lambda(lambda x: K.squeeze(x, 1))(self.conv_7)
         
        # bidirectional LSTM layers with units=128
        self.blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(self.squeezed)
        self.blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(self.blstm_1)
         
        self.outputs = Dense(len(char_list)+1, activation = 'softmax')(self.blstm_2)
        
        # model to be used at test time
        self.act_model = Model(self.inputs, self.outputs)
        
       # self.act_model.summary()
        
        self.act_model.load_weights('best_model.hdf5')

    def detect(self,x):
        img = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
        valid_img = []
        w,h = img.shape
        if h > 128 or w > 32:
            pass
        if w < 32:
            add_zeros = np.ones((32-w, h))*255
            img = np.concatenate((img, add_zeros))
         
        if h < 128:
            add_zeros = np.ones((32, 128-h))*255
            img = np.concatenate((img, add_zeros), axis=1)
        img = np.expand_dims(img , axis = 2)
        img = img/255.
        valid_img.append(img)
        valid_img = np.array(valid_img)
        self.xyz = valid_img
        #print(valid_img)
        #valid_img = np.reshape(valid_img,(1,32,128,1))
        prediction = self.act_model.predict(valid_img)
         
        # use CTC decoder
        out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],greedy=True)[0][0])
         
        # see the results
        i = 0
        result=""
        for x in out:
            for p in x:  
                if int(p) != -1:
                    result+=char_list[int(p)]       
        return result
            

text_detect = TextDetector()
text_detect.detect(cv2.imread('90kDICT32px/1/4/333_rezones_65882.jpg'))
# path = '90kDICT32px/'
# valid_img = []
# valid_txt = []
# valid_input_length = []
# valid_label_length = []
# valid_orig_txt = []
 
# max_label_len = 0
 
# i =1 
# flag = 0
# x = cv2.imread('90kDICT32px/1/4/333_rezones_65882.jpg')
# print(x)
# img = cv2.cvtColor( x , cv2.COLOR_BGR2GRAY )
# w,h = img.shape
# if h > 128 or w > 32:
#     pass
# if w < 32:
#     add_zeros = np.ones((32-w, h))*255
#     img = np.concatenate((img, add_zeros))
 
# if h < 128:
#     add_zeros = np.ones((32, 128-h))*255
#     img = np.concatenate((img, add_zeros), axis=1)
# img = np.expand_dims(img , axis = 2)

# # Normalize each image
# img = img/255.

# # get the text from the image
# txt = '333_rezones_65882'.split('_')[1]
# #txt = f_name.split('_')[1]
        
# # compute maximum length of the text
# if len(txt) > max_label_len:
#     max_label_len = len(txt)

# valid_orig_txt.append(txt)   
# valid_label_length.append(len(txt))
# valid_input_length.append(31)
# valid_img.append(img)
# valid_txt.append(encode_to_labels(txt))

# inputs = Input(shape=(32,128,1))
 
# # convolution layer with kernel size (3,3)
# conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
# # poolig layer with kernel size (2,2)
# pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
 
# conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
# pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
 
# conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
 
# conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
# # poolig layer with kernel size (2,1)
# pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
 
# conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
# # Batch normalization layer
# batch_norm_5 = BatchNormalization()(conv_5)
 
# conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
# batch_norm_6 = BatchNormalization()(conv_6)
# pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
 
# conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
 
# squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
 
# # bidirectional LSTM layers with units=128
# blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
# blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
 
# outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

# # model to be used at test time
# act_model = Model(inputs, outputs)

# act_model.summary()

# act_model.load_weights('best_model.hdf5')

# valid_img = np.reshape(valid_img,(1,32,128,1))

# prediction = act_model.predict(valid_img)
 
# # use CTC decoder
# out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],greedy=True)[0][0])
 
# # see the results
# i = 0
# for x in out:
#     print("original_text =  ", valid_orig_txt[i])
#     print("predicted text = ", end = '')
#     for p in x:  
#         if int(p) != -1:
#             print(char_list[int(p)], end = '')       
#     print('\n')
#     i+=1