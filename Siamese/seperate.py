import numpy as np
import math
import os
import csv
from natsort import natsorted
import shutil


def saperate_train_valid(y,action):
    trainxNames = []
    validxNames = []
    c = []
    
    tempdata = [i for i in range(len(y)) if y[i][action+1] == '1.0']
    
    for i in range(10):
        c.append(np.random.choice(tempdata))
    c = natsorted(c)


    for i in range(len(y)):
        if i in c:
            validxNames.append(y[i][0])
        elif i in tempdata:
            trainxNames.append(y[i][0])
            
    return trainxNames, validxNames

if __name__ == '__main__':
    dataset = 'ZPCTrain224'
    truths = 'ISIC2018_Task3_Training_GroundTruth\\ISIC2018_Task3_Training_GroundTruth.csv'
    trainpath = 'TrainSeamea'
    mempath = 'Memories'

    if not os.path.exists(mempath):
        os.makedirs(mempath)
    if not os.path.exists(trainpath):
        os.makedirs(trainpath)

    y= []
    with open (truths, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for line in reader:
            y.append(line)

    for i in range(7):
        if not os.path.isdir(os.path.join(trainpath,str(i))):
            os.mkdir(os.path.join(trainpath,str(i)))
        if not os.path.isdir(os.path.join(mempath,str(i))):
            os.mkdir(os.path.join(mempath,str(i)))
        
        trainxNames, validxNames = saperate_train_valid(y,i)
        print(len(trainxNames)+len(validxNames))
        for item in os.listdir(dataset):
            if item.endswith('.npy'):
                if item.replace('.npy','') in trainxNames:
                    shutil.copy2(os.path.join(dataset,item),os.path.join(trainpath,str(i),item))
                    #os.rename(os.path.join(dataset,item),os.path.join(trainpath,str(i),item))
                elif item.replace('.npy','') in validxNames:
                    shutil.copy2(os.path.join(dataset,item),os.path.join(mempath,str(i),item))
                    #os.rename(os.path.join(dataset,item),os.path.join(mempath,str(i),item))

        
