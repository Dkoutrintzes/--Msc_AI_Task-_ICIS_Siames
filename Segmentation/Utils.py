import numpy as np
import math
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img,array_to_img
import os

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

def compare(a,b):
    #print(a.shape)
    #print(b.shape)
    if a.shape == b.shape:
        N = 0
        T = 0
        for i in range(len(a)):
            for j in range(len(a[0])):
                #if b[i][j] == 1:

                    #print(a[i][j] , b[i][j])
                if a[i][j] > 0.5:
                    d = 1
                else:
                    d = 0
                if b[i][j] == d:
                    T += 1
                N += 1
        return T/N
    else:
        return False
dataset = 'E:\MSc\ISIC2018_Task3_Validation_Input'
def zeropadd(a,name):
    b = getimage(os.path.join(dataset,name.replace('npy','jpg')))
    xn = np.shape(b)[0]
    yn = np.shape(b)[1]

    a = cv2.resize(a, (xn, yn))
    #print(a.shape)
    #print(b.shape)
    if True:
        N = 0
        T = 0
        for i in range(len(a)):
            for j in range(len(a[0])):
                #if b[i][j] == 1:

                    #print(a[i][j] , b[i][j])
                if a[i][j] > 0.5:
                    d = 1
                else:
                    d = 0
                if d == 0:
                    #print(b[i][j])
                    b[i][j] = [0,0,0]
        b = corp_image(b)         
        return b
    else:
        return False



def corp_image(image):
    minx = 50000
    maxx = 0
    miny = 50000
    maxy = 0

    for  x in range(len(image)):
        for y in range(len(image[0])):
            #print(image[x][y])
            if any(image[x][y] != [0,0,0]):
                if minx > x:
                    minx = x
                if miny > y:
                    miny = y
            if any(image[x][y] != [0,0,0]):
                if maxx < x:
                    maxx = x
                if maxy < y:
                    maxy = y
    #print(maxx,minx,maxy,miny)
    if minx > maxx and miny > maxy:
        
        newimage = cv2.resize(image, (224, 224))
    elif minx ==  maxx  and maxy == miny:
        newimage = cv2.resize(image, (224, 224))
    else:
        newimage = np.zeros([maxx-minx,maxy-miny,3],dtype='float32')
        #print(np.shape(newimage))
        i = 0
        
        for x in range(minx,maxx,1):
            u = 0
            for y in range(miny,maxy,1):
                #print(i,u)
                newimage[i][u] = image[x][y]
                u += 1
            i += 1
        newimage = cv2.resize(newimage, (224, 224))
    return newimage