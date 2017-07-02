# -*- coding: utf-8 -*-
"""
This is the testing code of "Vehicle Detection and Classification for Traffic Surveillance Systems"
The testing data is STL-10
author: Jianlong Chang
school: Department of National Laboratory of Pattern Recognition, Institute of Automation, Chinese Academy of Sciences
e-mail: jianlong.chang@nlpr.ia.ac.cn
"""

from __future__ import absolute_import
from __future__ import print_function
import scipy.io as sio
from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense,MaxPooling2D,Convolution2D,Highway
from keras.layers import Dropout,Flatten,Input,BatchNormalization
from keras import backend as K
import theano.tensor as T
from keras.utils import np_utils
from keras.engine.topology import Layer,InputSpec
from keras.regularizers import l2, activity_l2,activity_l1
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.models import model_from_json 
import random
import h5py

np.random.seed(12345)
global datagen
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1
)
#The setting of proposed DNN
img_channels,img_rows,img_cols = 1,64,64
n_ker,n_channels,w_ker,l_ker = 25,1,11,11
nb_classes = 10
batch_size = 128
nb_training = 0.8
sparse_layer1 =0.5
sparse_layer2 = 0.5
sae_layer1 = 1024
sae_layer2 = 512

res = []

nb_labeled_data = [500,1000,1500,2000,2500]

matfn="labeled.mat"#labeled data
f = h5py.File(matfn)
X_train = f['X_train']
X_train = np.transpose(X_train, (3, 2, 1,0))
X_test = f['X_test']
X_test = np.transpose(X_test, (3, 2, 1,0))
Y_train = f['Y_train']
Y_train = np.transpose(Y_train, (1,0))
Y_test = f['Y_test']
Y_test = np.transpose(Y_test, (1,0))
Y_train = Y_train-1
Y_test = Y_test-1
Y_train = np_utils.to_categorical(Y_train,nb_classes)
Y_test = np_utils.to_categorical(Y_test,nb_classes)

for nb_data in nb_labeled_data:
    #utilizing different number of training data to train network
    inp_img = Input(shape = (img_channels,img_rows,img_cols))
    feature_map = Convolution2D(n_ker, w_ker, l_ker, 
                                activation = 'relu',
                                name = 'cnn_layer')(inp_img)
    pool_map = MaxPooling2D(pool_size=(4, 4),name = 'pool_layer')(feature_map)
    cnn_feature = Flatten()(pool_map)
    x1 = Dense(sae_layer1,activation = 'relu',name = 'sae1_layer')(cnn_feature)
    x2 = Dense(sae_layer2,activation = 'relu',name = 'sae2_layer')(x1)
    y = Dense(nb_classes,activation = 'softmax',name = 'classfier')(x2)
    model = Model(inp_img,y)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])
    # The training data in each step
    X = X_train[0:nb_data,:,:,:]
    Y = Y_train[0:nb_data,:]
    #load pre-trained parameters
    model.load_weights('sc_ae_weights.h5')
    cnn_layer = model.get_layer(name = 'cnn_layer')
    sae1_layer = model.get_layer(name = 'sae1_layer')
    sae2_layer = model.get_layer(name = 'sae2_layer')
    softmax = model.get_layer(name = 'classfier')
    # fixed other layer, train softmax only
    cnn_layer.trainable = False
    sae1_layer.trainable = False
    sae2_layer.trainable = False
    model.fit_generator(datagen.flow(X, Y,batch_size=batch_size),
                                  samples_per_epoch=4*X.shape[0], nb_epoch=100)
    score = model.evaluate(X_test,Y_test,batch_size = batch_size,verbose=0)
    
    #fintuning
    #fix convolutional feature learning part and update other parameters of DNN
    sae1_layer.trainable = True
    sae2_layer.trainable = True
    
    model.fit_generator(datagen.flow(X, Y,batch_size=batch_size),
                                  samples_per_epoch=4*X.shape[0], nb_epoch=100)
    score = model.evaluate(X_test,Y_test,batch_size = batch_size,verbose=0)
    #update all parameters of DNN
    cnn_layer.trainable = True
    model.fit_generator(datagen.flow(X, Y,batch_size=batch_size),
                                  samples_per_epoch=4*X.shape[0], nb_epoch=100)
    score = model.evaluate(X_test,Y_test,batch_size = batch_size,verbose=0)
    print(score)
    res.append(score[1])
    # recording the results
    fp = open('sc_ae.txt','w')
    for i in res:
        fp.write(str(i)+'\n')
    fp.close()
