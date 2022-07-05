import numpy as np
from tensorflow import  keras
import os
import csv
import random
import pathlib
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical,Sequence
import math
import time
import csv
import threading

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
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
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        dataset = 'E:\MSc\ZPCValid224'
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        y = np.empty((self.batch_size,7), dtype=int)
        dim = (224,224)
        X= np.empty((self.batch_size, *dim, self.n_channels))
        
        for i, ID in enumerate(list_IDs_temp):
            
            with open(os.path.join(dataset,ID+'.npy'),'rb') as f:
                X[i] = np.load(f)

                # Store class
                y[i] = self.labels[ID]

        return X, y

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
