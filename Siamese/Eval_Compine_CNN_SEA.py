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
from Seamea import build_seamea,build_seamea_cnn
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
    model = load_model('E:\MSc\ModelsCNN2\\07-01-2022_02-45-57\\15-0.26.h5')

    dataset = 'E:\MSc\ZPCValid'
    mem = 'Memories'
    truths = 'E:\MSc\ISIC2018_Task3_Validation_GroundTruth\ISIC2018_Task3_Validation_GroundTruth.csv'
    xNames = []
    yNames = []
    y = []

    sem_model = build_seamea_cnn((86,86,3))
    sem_model.load_weights('E:\MSc\SeaModel\WeightsZPc3\\39-0.0.h5')
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

    ythreepreds = []
    scores = []
    for i in predictions:
        temparr = []
        temp = []
        for t in range(3):
            tempmax = -1
            scoremax = -1
            for j in range(len(i)):
                if float(i[j]) > scoremax and j not in temparr:
                    scoremax = float(i[j])
                    tempmax = j
            temparr.append(tempmax)
            temp.append(scoremax)
        scores.append(temp)
        ythreepreds.append(temparr)

    ythreepreds = np.array(ythreepreds)
    #print(ythreepreds)
    #print(scores)
    tp3 = 0


    for i in range(len(ythreepreds)):
        print(ythreepreds[i])
        print(scores[i][0])
        if scores[i][0] > float(0.9):
            if int(ytruths[i]) == int(ythreepreds[i][0]):
                #print('hiii')
                tp3 += 1
        else:
            x_geuine_pair_1 = np.zeros([1,86,86,3],dtype='float32')
            maxid = -1
            mindist = 99999
            with open(os.path.join('E:\MSc\ZPCValid',xNames[i]+'.npy'),'rb') as f:
                        x_geuine_pair_1[0, :]= np.load(f)
            for j in ythreepreds[i]:
                avepermem = 0
                #print('here',j)
                for memory in os.listdir(os.path.join(mem, str(j))):
                    x_geuine_pair_2 = np.zeros([1,86,86,3],dtype='float32')
                        
                    with open(os.path.join('ZPCTrain',memory),'rb') as f:
                        x_geuine_pair_2[0, :] = np.load(f)
                    pred = sem_model.predict([x_geuine_pair_1,x_geuine_pair_2])
                    #print(i,'---',j,'---',pred[0])
                    #avepermem += float(pred[0])
                    #avepermem = avepermem/int(len(os.listdir(os.path.join(mem, str(j)))))
                    if mindist > float(pred[0]):
                        mindist = float(pred[0])
                        maxid = j
            print(ytruths[i],'---',maxid,'---',mindist)
            if int(ytruths[i]) == int(maxid):
                #print('hiii')
                tp3 += 1
    print(tp3/len(ythreepreds))
        

    


            
