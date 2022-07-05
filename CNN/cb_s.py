import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import datetime
# from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

#import files_location

base_dir='./'
#checkpoints_dir=files_location.checkpoints_dir


now = datetime.now()
date_time = now.strftime("%m-%d-%Y_%H-%M-%S")


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                ax.text(j, i, "   " + format(cm[i, j], fmt),
                        ha="left", va="center",
                        color="black")
    fig.tight_layout()


    #plt.show()
    plt.savefig("./data/plots/7seq45_batch8_cm.png")

    return ax


def checkpointer(checkpoints_dir):
    "Make the files with the checkpoint"
    filepath = os.path.join(checkpoints_dir, date_time)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    filename = os.path.join(filepath,
                            "{epoch:02d}-{loss:.2f}.h5")

    checkpoint = ModelCheckpoint(filepath=filename,
                                 save_best_only=False)

    return checkpoint


def stopper(patience):
    earlystop = EarlyStopping(monitor='loss', min_delta=0,
                              patience=patience,
                              verbose=1, mode='auto', baseline=None)

    return earlystop


def board():
    "Make the tenlogs"
    tensorlogs = os.path.join(base_dir, "tensorboard", date_time)
    if not os.path.exists(tensorlogs):
        os.makedirs(tensorlogs)

    tensorboard = TensorBoard(log_dir=tensorlogs, histogram_freq=0,
                              batch_size=32,
                              write_graph=True, write_grads=False,
                              write_images=True, embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None,
                              embeddings_data=None, update_freq='epoch')
    return(tensorboard)
