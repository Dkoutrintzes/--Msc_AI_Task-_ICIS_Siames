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



def resize(input_image, input_mask):
   input_image = tf.image.resize(input_image, (128, 128), method="nearest")
   input_mask = tf.image.resize(input_mask, (128, 128), method="nearest")
   return input_image, input_mask

def augment(input_image, input_mask):
   if tf.random.uniform(()) > 0.5:
       # Random flipping of the image and mask
       input_image = tf.image.flip_left_right(input_image)
       input_mask = tf.image.flip_left_right(input_mask)
   return input_image, input_mask


def normalize(input_image, input_mask):
   input_image = tf.cast(input_image, tf.float32) / 255.0
   input_mask -= 1
   return input_image, input_mask

def load_image_train(datapoint):
   input_image = datapoint["image"]
   input_mask = datapoint["segmentation_mask"]
   input_image, input_mask = resize(input_image, input_mask)
   input_image, input_mask = augment(input_image, input_mask)
   input_image, input_mask = normalize(input_image, input_mask)
   return input_image, input_mask


def load_image_test(datapoint):
   input_image = datapoint["image"]
   input_mask = datapoint["segmentation_mask"]
   input_image, input_mask = resize(input_image, input_mask)
   input_image, input_mask = normalize(input_image, input_mask)
   return input_image, input_mask

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "True Mask", "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
        plt.show()





def create_mask(pred_mask):
 pred_mask = tf.argmax(pred_mask, axis=-1)
 pred_mask = pred_mask[..., tf.newaxis]
 return pred_mask[0]
def show_predictions(dataset=None, num=1):
 if dataset:
   for image, mask in dataset.take(num):
     pred_mask = unet_model.predict(image)
     display([image[0], mask[0], create_mask(pred_mask)])
 else:
   display([sample_image, sample_mask,
            create_mask(model.predict(sample_image[tf.newaxis, ...]))])

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

   dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

   train_dataset = dataset["train"].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
   print(train_dataset)
   # test_dataset = dataset["test"].map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

   # BATCH_SIZE = 8
   # BUFFER_SIZE = 1000
   # train_batches = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
   # train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
   # validation_batches = test_dataset.take(3000).batch(BATCH_SIZE)
   # test_batches = test_dataset.skip(3000).take(669).batch(BATCH_SIZE)


   # sample_batch = next(iter(train_batches))
   # random_index = np.random.choice(sample_batch[0].shape[0])
   # sample_image, sample_mask = sample_batch[0][random_index], sample_batch[1][random_index]
   # display([sample_image, sample_mask])

   # unet_model = build_unet_model()
   # unet_model.compile(optimizer = tf.keras.optimizers.Adam(1e-4), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
   # unet_model.summary()
   
   # NUM_EPOCHS = 5
   # TRAIN_LENGTH = info.splits["train"].num_examples
   # STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
   # VAL_SUBSPLITS = 5
   # TEST_LENTH = info.splits["test"].num_examples
   # VALIDATION_STEPS = TEST_LENTH // BATCH_SIZE // VAL_SUBSPLITS
   # model_history = unet_model.fit(train_batches,
   #                             epochs=NUM_EPOCHS,
   #                             steps_per_epoch=STEPS_PER_EPOCH,
   #                             validation_steps=VALIDATION_STEPS,
   #                             validation_data=test_batches)
   
   # count = 0
   # for i in test_batches:
   #     count +=1
   # print("number of batches:", count)

   unet = build_unet_model()
   #unet = load_model("unet_model.h5")
   unet.compile(optimizer = tf.keras.optimizers.Adam(1e-4), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
   unet.summary()
   

   dataset = 'E:\MSc\TrainDataNP256'
   truths = 'E:\MSc\TrainDataTruths256'


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
   
   trainxNames, trainy, validxNames, validy = saperate_train_valid(xNames, y)

   Ytrain = {trainxNames[i]: trainy[i] for i in range(len(trainy))}
   Yvalid = {validxNames[i]: validy[i] for i in range(len(validy))}
      
   train_generator = load.DataGenerator(trainxNames, Ytrain, batch_size=batch, n_channels=3, shuffle=True)
   valid_generator = load.DataGenerator(validxNames, Yvalid, batch_size=batch, n_channels=3, shuffle=True)

   # # Callbacks
   cp = cb_s.checkpointer()

   # st= cb_s.stopper(30)
   csv_logger = CSVLogger('./a.log', separator=',', append=False) #store the entire training history in ./training.log

   earlystoper = cb_s.stopper(10)

   lr = lrclass(len(trainy)//batch,30)

   lrate = LearningRateScheduler(lr.lr_polynomial_decay,len(trainy)//batch)
   #tb = TensorBoard(log_dir=os.path.join('Models', 'logs', '{}'.format(time())), histogram_freq=0,write_graph=True)

   h=unet.fit_generator(generator=train_generator,validation_data=valid_generator,steps_per_epoch=len(trainy)//batch,validation_steps=len(validy)//batch,
                        epochs=100,
                        verbose=1,use_multiprocessing=False, workers=0,
                        max_queue_size=1,callbacks=[earlystoper, cp,lrate, csv_logger])
   








    

    













