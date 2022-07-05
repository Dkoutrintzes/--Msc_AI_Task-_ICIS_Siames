import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import load_model
import time
import os
import cb_s
import csv
import math 
import CnnGenerator as load
from Unn import build_unet_model
from natsort import natsorted
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
import time

config = tf.compat.v1.ConfigProto()
#config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)



def saperate_train_valid(y,actions):
    dataset = 'E:\MSc\ZPCTrain224'
    trainxNames = []
    trainy = []
    tempdata = []
    validxNames = []
    validy = []
    c = []
    for label in range(0, actions):
        tempdata = [i for i in range(len(y)) if y[i][label+1] == '1.0']
        e = int(len(tempdata) * 0.1)
        for i in range(e):
            c.append(np.random.choice(tempdata))
    c = natsorted(c)
    for i in range(len(y)):
        #print()
        if os.path.isfile(os.path.join(dataset,y[i][0]+'.npy')):
            if i in c:
                #print(y[i][0],y[i])
                validxNames.append(y[i][0])
                temp = y[i][1:]
                for j in range(len(temp)):
                    temp[j] = int(float(temp[j]))
                validy.append(temp)
            else:
                #print(y[i][0],y[i])
                temp = y[i][1:]
                for j in range(len(temp)):
                    temp[j] = int(float(temp[j]))
                trainxNames.append(y[i][0])
                trainy.append(temp)
    return trainxNames, trainy, validxNames, validy




epoch = 200
batch = 16





def step_decay( epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

class lrclass:
    def __init__(self,steps_per_epoch,epoch):
        self.steps_per_epoch = steps_per_epoch
        self.epoch = epoch
    def lr_polynomial_decay(self,global_step):
        learning_rate = 0.000005
        end_learning_rate = 0.0000003
        decay_steps = self.steps_per_epoch * self.epoch
        power = 0.9
        p = float(global_step) / float(decay_steps)
        lr = (learning_rate - end_learning_rate) * np.power(1 - p, power) + end_learning_rate
        return lr



if __name__ == '__main__':
    
    model = VGG16(include_top=False,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=(224, 224, 3),
                      pooling=None,
                      classes=7)
    model.summary()

    output = model.get_layer('block5_pool').output
    output = Flatten()(output)
    output = Dropout(0.5)(output)
    # output = Dense(512, activation='relu', name= 'Dense_layer',kernel_initializer='random_normal', kernel_regularizer=l1_l2(l1 = 0.0001, l2 = 0.001 ))(output)
    output = Dense(256, activation='relu', name='feature', kernel_initializer='random_normal')(output)
    output = Dense(7, activation='softmax', name='fc1000')(output)
    new_model = Model(model.input, output)
    new_model.summary()
    new_model.compile(optimizer = tf.keras.optimizers.Adam(1e-4), loss = "categorical_crossentropy", metrics = ["accuracy"])
    dataset = 'E:\MSc\ZPCTrain224'
    truths = 'E:\MSc\ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth.csv'
    xNames = []
    y = []

    batch = 8

    for item in os.listdir(dataset):
        if item.endswith('.npy'):            
            xNames.append(os.path.join(dataset,item))

    with open (truths, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for line in reader:
            y.append(line)
    
    trainxNames, trainy, validxNames, validy = saperate_train_valid(y,7)
    print(len(validxNames),' Validation Data')
    Ytrain = {trainxNames[i]: trainy[i] for i in range(len(trainy)) }
    Yvalid = {validxNames[i]: validy[i] for i in range(len(validy)) }


    train_generator = load.DataGenerator(trainxNames, Ytrain, batch_size=batch, n_channels=3, shuffle=True)
    valid_generator = load.DataGenerator(validxNames, Yvalid, batch_size=batch, n_channels=3, shuffle=True)

    # # Callbacks
    cp = cb_s.checkpointer('ModelsCNN224')

    # st= cb_s.stopper(30)
    csv_logger = CSVLogger('./a.log', separator=',', append=False) #store the entire training history in ./training.log

    earlystoper = cb_s.stopper(10)

    lr = lrclass(len(trainy)//batch,30)

    lrate = LearningRateScheduler(lr.lr_polynomial_decay,len(trainy)//batch)
    #tb = TensorBoard(log_dir=os.path.join('Models', 'logs', '{}'.format(time())), histogram_freq=0,write_graph=True)

    h=new_model.fit_generator(generator=train_generator,validation_data=valid_generator,steps_per_epoch=len(trainy)//batch,validation_steps=len(validy)//batch,
                            epochs=100,
                            verbose=1,use_multiprocessing=False, workers=0,
                            max_queue_size=1,callbacks=[earlystoper, cp,lrate, csv_logger])