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


if __name__ == '__main__':
    model = load_model('E:\MSc\ModelsCNN224\\07-02-2022_19-20-11\\27-0.26.h5')

    dataset = 'E:\MSc\ZPCValid224'
    truths = 'E:\MSc\ISIC2018_Task3_Validation_GroundTruth\ISIC2018_Task3_Validation_GroundTruth.csv'
    xNames = []
    yNames = []
    y = []

    batch = 8

    

    with open (truths, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for line in reader:
            y.append(line)
    
    for line in y:
        xNames.append(line[0])
        temp = line[1:]
        for j in range(len(temp)):
            temp[j] = int(float(temp[j]))
        yNames.append(temp)
        


    Ytest = {xNames[i]:yNames[i] for i in range(len(xNames))}

    test_generator = load.DataGenerator(xNames,Ytest, batch_size=batch, n_channels=3, shuffle=False)

    test_loss, test_acc = model.evaluate_generator(test_generator, verbose=1)

    print(test_acc)

    predictions = model.predict_generator(test_generator, verbose=1)
    ypreds = []
    for i in predictions:
        tempmax = -1
        scoremax = -1.0
        for j in range(len(i)):
            if float(i[j]) > scoremax:
                scoremax = float(i[j])
                tempmax = j
        ypreds.append(tempmax)
    ypreds = np.array(ypreds)

    ytruths = []
    for i in yNames:
        tempmax = -1
        scoremax = -1.0
        for j in range(len(i)):
            if float(i[j]) > scoremax:
                scoremax = float(i[j])
                tempmax = j
        ytruths.append(tempmax)
    ytruths = np.array(ytruths)
    tp = 0
    for i in range(len(ypreds)):
        if int(ytruths[i]) == int(ypreds[i]):
            tp += 1
    print(tp/len(ypreds))

    ythreepreds = []
    for i in predictions:
        temparr = []
        for t in range(3):
            tempmax = -1
            scoremax = -1
            for j in range(len(i)):
                if float(i[j]) > scoremax and j not in temparr:
                    scoremax = float(i[j])
                    tempmax = j
            temparr.append(tempmax)
        ythreepreds.append(temparr)

    ythreepreds = np.array(ythreepreds)
    tp3 = 0
    for i in range(len(ythreepreds)):
        if ytruths[i] in ythreepreds[i]:
            tp3 += 1
    print(tp3/len(ythreepreds))
        

    


            
