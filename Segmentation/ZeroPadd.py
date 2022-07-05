from hashlib import new
from tkinter import X
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
from Utils import compare,zeropadd

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
    

    dataset = 'E:\MSc\ValidClassification'
    newdataset = 'E:\MSc\ZPCValid224'
    try:
        os.mkdir(newdataset)
    except:
        print("Directory already exists")


    xNames = []
    y = []

    batch = 8
    for item in os.listdir(dataset):
        #ISIC_0000000_segmentation.png
        #ISIC_0000000.jpg

        if item.endswith('.npy'):
            xNames.append(item)
                
    # Ytest = {xNames[i]: y[i] for i in range(len(y))}
        
    # tesr_generator = load.DataGenerator(xNames, Ytest, batch_size=batch, n_channels=3, shuffle=False)

    

    # predicts = unet.predict(tesr_generator)
    
    # items = os.listdir(dataset)

    for i in range(len(xNames)):
        if os.path.isfile(os.path.join(newdataset,xNames[i])) == False:
            print(xNames[i])
            with open(os.path.join(dataset,xNames[i]),'rb') as f:
                    treutemp = np.load(f)
                    treutemp = treutemp.reshape(1,128,128,3)
                    #print(np.shape(treutemp))

            predict = unet.predict(treutemp)

            predimage = rgb2gray(predict[0])

            newimage = zeropadd(predimage,xNames[i])
            print(newimage)
            with open(os.path.join(newdataset,xNames[i]),'wb') as f:
                    np.save(f,newimage)

            # from matplotlib import pyplot as plt

            # # Set the figure size
            # plt.rcParams["figure.autolayout"] = True

            # # Plot the data using imshow with gray colormap
            # plt.imshow(newimage, cmap='gray')

            # # Display the plot
            # plt.show()
            # break
        else:
            print('exist')



    
   








    

    













