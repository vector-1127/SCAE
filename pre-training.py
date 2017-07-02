# -*- coding: utf-8 -*-
"""
This is the pre-training code of "Vehicle Detection and Classification for Traffic Surveillance Systems"
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
from keras.regularizers import activity_l1,activity_l2
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.models import model_from_json 
import random
import h5py

global datagen
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
#The setting of proposed DNN
img_channels,img_rows,img_cols = 1,64,64
n_ker,n_channels,w_ker,l_ker = 25,1,11,11
nb_classes = 10
batch_size = 128
sae_layer1 = 1024
sae_layer2 = 512

# The structure of proposed DNN
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

cnn_layer = model.get_layer(name = 'cnn_layer')
sae1_layer = model.get_layer(name = 'sae1_layer')
sae2_layer = model.get_layer(name = 'sae2_layer')
softmax = model.get_layer(name = 'classfier')

#pre-training

#CNN feature learning
cnn_feature_learning_model = Model(inp_img,cnn_feature)
matfn="kernels_121_25.mat"# The kernels learned by sparse coding
data=sio.loadmat(matfn)
kernels=data['kernels']
w = cnn_layer.get_weights()
cnn_layer.set_weights([kernels,w[1]])# Initializing learned parameters to kernels

#SAE feature learning
#SAE (Stacked autoencoder)
matfn="unlabeled_data.mat"# unlabeled data of STL-10
f = h5py.File(matfn)
unlabeled_data = f['unlabeled_data']
unlabeled_data = np.transpose(unlabeled_data, (3, 2, 1,0))
cnn_features = cnn_feature_learning_model.predict(unlabeled_data)
unlabeled_data = None
# end-to-end structure of SAE
img_cnn_feature = Input(shape = (cnn_features.shape[1],))
encoded1 = Dense(sae_layer1,activation = 'relu')(img_cnn_feature)
encoded2 = Dense(sae_layer2,activation = 'relu')(encoded1)
decoded2 = Dense(sae_layer1,activation = 'relu')(encoded2)
decoded1 = Dense(cnn_features.shape[1],activation='sigmoid')(decoded2)

sae = Model(img_cnn_feature,decoded1)
sae.compile(optimizer='rmsprop', loss='mse')# The objective function of SAE
sae.fit(cnn_features,cnn_features,batch_size = batch_size,nb_epoch = 100)


#initialize
sae1_layer.set_weights(sae.layers[1].get_weights())
sae2_layer.set_weights(sae.layers[2].get_weights())
model.save_weights('sc_ae_weights.h5')

#saved parameters will employed to initialized proposed DNN
