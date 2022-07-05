import numpy as np
from tensorflow import  keras
import os
import csv
import random
import pathlib
from tensorflow.keras.preprocessing.image import img_to_array, load_img,array_to_img
from tensorflow.keras.utils import to_categorical,Sequence
import math
import time

import cv2
import threading
import re

dataset = 'E:\MSc\TrainClassification'
truelbl = 'E:\MSc\ValidDataTruths'
def getimage(path):
    image = load_img(os.path.join(path))
    #print(type(image))
    
    img_arr = img_to_array(image)
    #print(type(img_arr))
    img_arr = cv2.resize(img_arr,dsize=(128, 128) )
    x = (img_arr / 255.).astype(np.float32)

    where_are_NaNs = np.isnan(x)
    x[where_are_NaNs] = 0
    #print(np.shape(x))
    return x

def getimagegs(path):
    image = load_img(os.path.join(path))
    #print(type(image))
     
    img_arr = img_to_array(image.convert('L'))
    #print(type(img_arr))
    img_arr = cv2.resize(img_arr, dsize=(128, 128))
    x = (img_arr / 255.).astype(np.float32)

    where_are_NaNs = np.isnan(x)
    x[where_are_NaNs] = 0
    #print(np.shape(x))
    return x

def getnp(path):
    with open(os.path.join(MosaicDatasetnp,path), 'rb') as f:
        A = np.load(f)
        B = np.load(f)
        C = np.load(f)
        D = np.load(f)
        E = np.load(f)
    return A,B,C,D,E

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=32, n_channels=1, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        dim = (128,128)
        X= np.empty((self.batch_size, *dim, self.n_channels))
        


        y = np.empty((self.batch_size, *dim))
        #print(list_IDs_temp)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            with open(os.path.join(dataset,ID),'rb') as f:
                X[i] = np.load(f)
            # with open(os.path.join(truelbl,self.labels[ID]),'rb') as f:
            #     y[i] = np.load(f)
            
            # X[i] = getimage(os.path.join(dataset,ID))
            # y[i] = getimagegs(os.path.join(truelbl,self.labels[ID]))
            

       
        return X

def writeFile(data, labels, savePath):
    with open(savePath, mode='w',newline='') as data_files_paths:
        pic_writer = csv.writer(data_files_paths, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for pic, label in zip(data, labels):
            pic_writer.writerow([pic, label])


# read data from file
def readFile(readFilePath):
    data = []
    labels = []
    with open(readFilePath, mode='r') as data_files_paths:
        pic_reader = csv.reader(data_files_paths, delimiter=',', quotechar='"')
        for pic in pic_reader:
            data.append(pic[0])
            labels.append(pic[1])
    labels = np.asarray(labels)
    return data, labels
