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
import SeaGenerator as load
from Unn import build_unet_model
from Seamea import build_seamea,build_seamea_cnn
from natsort import natsorted
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from sklearn.utils import shuffle
import time

config = tf.compat.v1.ConfigProto()
#config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

def saperate_train_valid(xName,y):
    trainxNames = []
    trainy = []
    tempdata = []
    validxNames = []
    validy = []
    c = []
    for label in range(2):
        tempdata = [i for i in range(len(y)) if y[i] == label]
        e = int(len(tempdata) * 0.1)
        for i in range(e):
            c.append(np.random.choice(tempdata))
    c = natsorted(c)
    for i in range(len(y)):
        if i in c:
            #print(y[i][0],y[i])
            validxNames.append(xName[i])
            
            validy.append(y[i])
        else:
            #print(y[i][0],y[i])
            
            trainxNames.append(xName[i])
            trainy.append(y[i])
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
    dataset = 'TrainSeamea'
    mem = 'Memories'
    xNames = []
    
    count = 0 
    y = []
    t = 0
    step = 300
    for i in range(7):
        names = os.listdir(os.path.join(mem, str(i)))
        for name in os.listdir(os.path.join(dataset, str(i))):
            if t == step: 
                step+=500
                break
            for _ in range(7):
                if t == step: 
                    break
                t += 1
        for name in os.listdir(os.path.join(dataset, str(i))):
            if t == step: 
                step+=300
                break
            t += 1

    x_geuine_pair = np.zeros([t, 2,224,224,3],dtype='float32')
    y_genuine = np.zeros([t, 1])

    step = 300
    for i in range(7):
        names = os.listdir(os.path.join(mem, str(i)))
        #if count == 4999: break
        for name in os.listdir(os.path.join(dataset, str(i))):
            
            if count == step: 
                step+=500
                break
            for _ in range(5):
                if count == step: 
                    break
                choise = np.random.choice(names,1)
                choise = choise[0]
                #choise = choise.replace('.npy','')

                # xNames.append(choise+'--'+name.replace('.npy',''))
                with open(os.path.join('ZPCTrain224',choise),'rb') as f:
                    #print(np.shape(np.load(f)))
                    x_geuine_pair[count, 0, :] = np.load(f)
                with open(os.path.join('ZPCTrain224',name),'rb') as f2:
                    #print(np.shape(np.load(f2)))
                    a = np.load(f2)
                    x_geuine_pair[count, 1, :] = a

                y_genuine[count] = 1
                count += 1
                #print(count)
        
        for name in os.listdir(os.path.join(dataset, str(i))):
            
            if count == step: 
                step+=300
                break
            file = i
            while file == i:
                file = np.random.choice(7,1)
            #print(i, file)
            fnames = os.listdir(os.path.join(mem, str(file[0])))
            
            choise = np.random.choice(fnames,1)
            choise = choise[0]
            #choise = choise.replace('.npy','')
            #xNames.append(choise+'--'+name.replace('.npy',''))

            with open(os.path.join('ZPCTrain224',choise),'rb') as f:
                x_geuine_pair[count, 0, :]= np.load(f)
            with open(os.path.join('ZPCTrain224',name),'rb') as f:
                x_geuine_pair[count, 1, :] = np.load(f)
            
            y_genuine[count] = 0
            count += 1
        

            
    print(count,t)
    # print(x_geuine_pair[0, 0, :])
    # print(x_geuine_pair[0, 1, :])

    from matplotlib import pyplot as plt

    # Set the figure size
    plt.rcParams["figure.autolayout"] = True

    # Plot the data using imshow with gray colormap
    plt.imshow(x_geuine_pair[0,0,:], cmap='gray')

    # Display the plot
    plt.show()

    # trainxNames, trainy, validxNames, validy = saperate_train_valid(xNames,y)

    # Ytrain = {trainxNames[i]: trainy[i] for i in range(len(trainy))}
    # Yvalid = {validxNames[i]: validy[i] for i in range(len(validy))}

    # train_generator = load.DataGenerator(trainxNames, Ytrain, batch_size=batch, n_channels=3, shuffle=True)
    # valid_generator = load.DataGenerator(validxNames, Yvalid, batch_size=batch, n_channels=3, shuffle=True)

    # model = build_seamea(256)
    # # # Callbacks
    # cp = cb_s.checkpointer('ModelsSea')

    # # st= cb_s.stopper(30)
    # csv_logger = CSVLogger('./a.log', separator=',', append=False) #store the entire training history in ./training.log

    # earlystoper = cb_s.stopper(10)

    # lr = lrclass(len(trainy)//batch,30)

    # lrate = LearningRateScheduler(lr.lr_polynomial_decay,len(trainy)//batch)
    # #tb = TensorBoard(log_dir=os.path.join('Models', 'logs', '{}'.format(time())), histogram_freq=0,write_graph=True)

    # h=model.fit_generator(generator=train_generator,validation_data=valid_generator,steps_per_epoch=len(trainy)//batch,validation_steps=len(validy)//batch,
    #                         epochs=100,
    #                         verbose=1,use_multiprocessing=False, workers=0,
    #                         max_queue_size=1,callbacks=[earlystoper, cp,lrate, csv_logger])

    earlyStopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=100,
                              verbose=1,
                              restore_best_weights=True)

    checkpointer = ModelCheckpoint(
            filepath=os.path.join( 'SeaModel\WeightsZP224\{epoch:02d}-{val_loss:.1f}.h5'),
            verbose=1, save_weigts_only=True)

    callback_early_stop_reduceLROnPlateau = [earlyStopping, checkpointer]
    
    x_geuine_pair, y_genuine = shuffle(x_geuine_pair, y_genuine)

    img_1 = x_geuine_pair[:, 0]
    img_2 = x_geuine_pair[:, 1]
    in_shape = np.shape(img_1[0])
    print(in_shape)
    model = build_seamea_cnn(in_shape)
    print(np.shape(img_1))
    print(np.shape(img_2))
    print(np.shape(y_genuine))
    history = model.fit([img_1, img_2], y_genuine, validation_split=.20,
                    batch_size=16, verbose=1, epochs=100, callbacks=callback_early_stop_reduceLROnPlateau)






