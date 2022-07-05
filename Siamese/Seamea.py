from __future__ import print_function
import os
import time
from typing import Sequence
import numpy as np
from numpy.core.fromnumeric import shape
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
import random
from tensorflow.keras.constraints import max_norm
import pathlib
import tensorflow as tf
import os
import time

import tensorflow.keras as  keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers,optimizers
from tensorflow.keras.constraints import max_norm
import warnings
from tensorflow.keras.layers import Input,Lambda
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

from tensorflow.keras.models import load_model
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()




def build_seamea(input):
    Input1 = Input(shape=input)
    Input2 = Input(shape=input)

    x = Dense(units=128, activation='relu')

    x1 = x(Input1)
    x2 = x(Input2)

    # x1 = Dropout(0.5)(x1)
    # x2 = Dropout(0.5)(x2)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([x1, x2])

    #output = layers.Dense(1, activation='sigmoid')(distance)
    #norm = layers.BatchNormalization()(output)
    siamese_net = Model(inputs=[Input1, Input2], outputs=distance)

    # # Add a customized layer to compute the absolute difference between the encodings
    # L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    # L1_distance = L1_layer([x1,x2])
    
    # # Add a dense layer with a sigmoid unit to generate the similarity score
    # prediction = Dense(1,activation='sigmoid')(L1_distance)
    
    # # Connect the inputs with the outputs
    # siamese_net = Model(inputs=[Input1,Input2],outputs=prediction)

    #rms = keras.optimizers.SGD(momentum=0.9, decay=0)
    #rms = optimizers.Adam()
    rms = optimizers.Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #rms = optimizers.RMSprop()
    #siamese_net = tf.keras.Model(inputs=[Input1, Input2], outputs=distance)
    siamese_net.summary()
    
    siamese_net.compile(loss=contrastive_loss, optimizer=rms,metrics=[accuracy])

    return siamese_net


def build_seamea_cnn(input):
    cnn = load_model('E:\MSc\ModelsCNN224\\07-02-2022_19-20-11\\27-0.26.h5')
    cnn.summary()
    features = Model(inputs = cnn.input,outputs = cnn.get_layer('feature').output)
    features.summary()
    #fine_tune_at = len(features.layers) - int(len(features.layers) * .10)
    #freeztrain = True
    for layer in features.layers:
        
        layer.trainable = False
        if layer.name == 'block4_pool':
            break
    
    Input1 = Input(shape=input)
    Input2 = Input(shape=input)

    model1 = features(Input1)
    model2 = features(Input2)

    

    #x = Dense(units=128, activation='relu')

    #x1 = tf.keras.layers.GlobalAveragePooling2D()(model1)
    #x2 = tf.keras.layers.GlobalAveragePooling2D()(model2)

    #outputs1 = tf.keras.layers.Dense(128, activation='relu')(x1)    
    #outputs2 = tf.keras.layers.Dense(128, activation='relu')(x2)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([model1, model2])


    siamese_net = Model(inputs=[Input1, Input2], outputs=distance)

    #rms = optimizers.Adam()
    rms = optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #rms = optimizers.RMSprop()
    #siamese_net = tf.keras.Model(inputs=[Input1, Input2], outputs=distance)
    siamese_net.summary()
    
    siamese_net.compile(loss=contrastive_loss, optimizer=rms,metrics=[accuracy])

    return siamese_net
#build_seamea_cnn((224,224,3))

