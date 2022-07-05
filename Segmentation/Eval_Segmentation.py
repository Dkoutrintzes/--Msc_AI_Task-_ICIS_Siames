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
import math 
import Ganerator as load
from Unn import build_unet_model
config = tf.compat.v1.ConfigProto()
#config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)
from Utils import compare

def saperate_train_valid(xNames, y):
   trainxNames = []
   trainy = []
   validxNames = []
   validy = []
   c = []
   t = int(len(xNames) * 0.1)


   for i in range(t):
      c.append(np.random.choice(xNames))
   
   for i in range(len(y)):
      if xNames[i] in c:
         validxNames.append(xNames[i])
         validy.append((y[i]))
      else:
         trainxNames.append(xNames[i])
         trainy.append((y[i]))
   return trainxNames, trainy, validxNames, validy
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
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


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "True Mask", "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
        plt.show()
if __name__ == '__main__':

   
    
    unet = load_model("E:\MSc\Models\\06-02-2022_20-16-04\\27-0.09.h5")
    unet.compile(optimizer = tf.keras.optimizers.Adam(1e-4), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
    unet.summary()
    

    dataset = 'E:\MSc\ValidDataNP'
    truths = 'E:\MSc\ValidDataTruths'


    xNames = []
    y = []

    batch = 8
    for item in os.listdir(dataset):
        #ISIC_0000000_segmentation.png
        #ISIC_0000000.jpg

        if item.endswith('.npy'):
            titem = item.replace('.npy', '_segmentation.npy')
            if os.path.isfile(os.path.join(truths,titem)):
                xNames.append(os.path.join(dataset,item))
                y.append(os.path.join(truths,titem))

    Ytest = {xNames[i]: y[i] for i in range(len(y))}
        
    tesr_generator = load.DataGenerator(xNames, Ytest, batch_size=batch, n_channels=3, shuffle=False)

    #print(unet.evaluate(tesr_generator))

    predicts = unet.predict(tesr_generator)
    tp = 0
    
    
    for i in range(len(predicts)):
        #print(predicts[i])
        
        with open(os.path.join(truths,y[i]),'rb') as f:
                treutemp = np.load(f)
        #plt.imshow(rgb2gray(predicts[i]))
        #plt.show()
        #display([rgb2gray(predicts[0])])
        
        if compare(rgb2gray(predicts[i]), treutemp) > 0.95:
            tp += 1
        else:
            print(compare(rgb2gray(predicts[i]), treutemp))
    print('hi')    
    print(tp/len(y))
    
   








    

    













