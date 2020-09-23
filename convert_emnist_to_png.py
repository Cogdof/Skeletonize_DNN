import pandas as pd
import numpy as np
import warnings
import cv2
import os
import gc
from PIL import Image

'''
# Convert EMNIST, MNIST, etc binary csv data to jpg image type.

# Dataset from 

# Train :  697931
# Test :  116322

# Structre
-----------------------------
Train | Test
    /1 / {num}_{label}.jpg .....
    /2 
    ...
    /A
    ...
    /a
-----------------------------    
'''



# mapping label to alphabet for digit
def mapping(a):
    if a <= 9:
        return a + 48
    elif 10 <= a and a <= 35:
        return a + 55
    elif 36 <= a :
        return a + 61



warnings.filterwarnings("ignore")

#train = pd.read_csv('/home/mll/v_mll3/OCR_data/dataset/MNIST_dataset/EMNIST_byclass/test.csv') # test csv case
train = pd.read_csv('/home/mll/v_mll3/OCR_data/dataset/MNIST_dataset/EMNIST_byclass/emnist-byclass-train.csv')
#test = pd.read_csv('/home/mll/v_mll3/OCR_data/dataset/MNIST_dataset/EMNIST_byclass/emnist-byclass-test.csv')

print(train.shape)
#print(test.shape)

train_dir =  '/home/mll/v_mll3/OCR_data/dataset/MNIST_dataset/EMNIST_byclass/Train/'
#test_dir = '/home/mll/v_mll3/OCR_data/dataset/MNIST_dataset/EMNIST_byclass/Test/'

i = 0
for x in train._get_values:


    label = chr(mapping(x[0]))
    vector = x[1:]
    #print(label)
    #print(vector)
    img = vector.reshape(28, 28).astype(int)
    path = train_dir +"/"+ label
    if not (os.path.isdir(path)):  # 새  파일들을 저장할 디렉토리를 생성
        os.makedirs(os.path.join(path))

    # transform to fit rotate, flip.

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.flip(img, 1)
    cv2.imwrite(path+"/{}_{}.jpg".format(i,label) , img)

    #print(i)
    i= i+1

    gc.collect()

gc.collect()

'''
i = 0
for x in test._get_values:


    label = chr(mapping(x[0]))
    vector = x[1:]
    img = vector.reshape(28, 28).astype(int)
    path = test_dir +"/"+ label
    if not (os.path.isdir(path)):  # 새  파일들을 저장할 디렉토리를 생성
        os.makedirs(os.path.join(path))

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.flip(img, 1)
    cv2.imwrite(path+"/{}_{}.jpg".format(i,label) , img)
    i = i+1
    gc.collect()


'''


# gc
gc.collect()

print("Done!")