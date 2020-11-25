import io

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image, ImageOps
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import PIL
import sys
import math
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score

import SKDNN.VGG19_47 as VGG19_47
import SKDNN.VGG19_62 as VGG19_62
import SKDNN.ConnNet_Autoencoder as ConnNet_Autoencoder
import SKDNN.ConnNet_DT as ConnNet_DT
import SKDNN.ConnNet_Linear as ConnNet_Linear
import SKDNN.ConnNet_SVM as ConnNet_SVM
from sklearn.metrics import classification_report

import pickle
from sklearn.externals import joblib





'''
===============================================================================================
Init 2020.09.28 mon

[Based network]
Pytorch VGG19

[Dataset] 
v1.x English single character set(external data)

v2.x skeletonized external data

v3.x Skeletonized_character_dataset6  : Recognition character ->  dataset

v4.x EMNIST OCR 62 label dataset *

ver 6.1x Simple network with original skeletonized dataset (crop from CRAFT, Deep-text data)
ver 6.xx Simple network with generated synth char dataset 

ver 7.0x Decision Tree

-----------------------------------------------------


Data rebuliding...
version sequence also change..



[Lastest update] : 2020.11.20

================[VGG19 version]================
ver 1.0 batch 8, epoch 5
ver 1.1 batch 8, epoch 10
ver 1.2 batch 8, epoch 20
ver 1.3 batch 4, epoch 5

ver 2.1 batch 8 epoch 5, skeletonize(external data)
ver 2.2 batch 8 epoch 10, skeletonize(external data)
ver 2.3 batch 4, epoch 10, skeletonize(external data)
ver 2.4 batch 4, epoch 5, skeletonize(external data), change f1,f2,f3 layers

ver 4.0 batch 8 epoch 10 VGG 26+26+10 case,digit of EMNIST dataset.
ver 4.1 batch 16 epoch 10
ver 4.2 batch 4  epoch 10 , resize 224 -> 28*28 784
ver 4.3 batch 4 epoch 10(test), resize244 to 784, working fc layer
ver 4.4 batch 4 epoch 20 resize 784, fc3
ver 4.5 batch 4 epoch 100 resize 224, fc3 -> to late
ver 4.6 batch 4 epoch 10 resize 224 balanced                                                                         : Valid % | Test %         [now training]
ver 4.7 batch 4 epoch 30 resize 224 balanced   

ver 5.3 VGG19 , batch 16, epoch 10 resize 224, Data : EMNIST balance
ver 5.4 VGG19,  batch 16, epoch 20, resize 224, label 47,   [Dataset:EMNIST_balance]                                 : Valid 94% | Test 88%

[EMNIST_Letter_vgg and spinalVGG.py]xxxxxxxxxxxxxx
ver 5.0 spinalnet + vgg5 with EMNIST byclass    
ver 5.1 spinalnet + vgg5 with EMNIST balance     Valid : 90 | test : 24 


================[ConnNet]======================

ver 6.x Simple network
ver 6.1x = 47 label
ver 6.xx = 62 label

ver 6.11 + VGG19 5.4_47 : simplenet epoch 30, batch16 [Dataset: skeletonized_character_Dataset_1021]                : Valid  2% | Test  2% 
ver 6.12 + VGG19 5.4_47 : simplenet epoch 20, batch16 [Dataset: skeletonized_character_Dataset_1021]                : Valid  2% | Test  1% 
ver 6.13 + VGG19 5.4_47 : simplenet epoch 100, batch16, net_conv2 [Dataset: skeletonized_character_Dataset_1021]    : Valid  2% | Test  1%
ver 6.14 + VGG19 5.4_47 : autoencoder epoch 30, batch16, net_conv2 [Dataset: skeletonized_character_Dataset_1021]   : Valid  2% | Test  1%
ver 6.15 + VGG19 5.4_47 : autoencoder epoch 10, batch16, net_conv2 [Dataset: skeletonized_character_Dataset_1021]   : Valid  2% | Test  2%
(simplenet -> linear)

Threshold :60%
ver 6.16 + VGG19 5.4_47 : linear epoch 10, batch16, net_conv2 [Dataset: test_char47]                                : Valid  2% | Test   1% 
ver 6.17 + VGG19 5.4_47 : linear epoch 30, batch16, net_conv2 [Dataset: EMNIST_balanced]                            : Valid 94 % | Test 86% 
ver 6.18 + VGG19 5.4_47 : linear, epoch 20, batch 16,  [Dataset :seperate_single_character (balance) ]              : Valid  59% | Test 57%

ver 6.21 + VGG19 5.4_47 : SVM, epoch 20, batch 16,  [Dataset :seperate_single_character (balance) ]                 : Valid   64% | Test 64%
ver 6.22 + VGG19 5.4_47 : SVM, epoch 20, batch 16,  [Dataset: EMNIST_balanced ]                                     : Valid  98 % | Test 87%    

ver 6.51 + VGG19 5.4_47 : SVM, epoch 20, batch 16,   [Dataset: generate_img ]                                        : Valid  97 % | Test 96 %   << over fitting?
                                               test  Dataset: test_char47                                       : Valid   % | Test  %   
                                               test  Dataset: EMNIST_balanced                                      : Valid  2% | Test 1%     >> ????
                                                     
ver 6.52 + VGG19 5.4_47 : SVM, epoch 20, batch 16,   [Dataset: EMNIST_balanced ]                                     : Valid  98 % | Test 87 %
ver 6.53 + VGG19 5.4_47 : SVM, epoch 20, batch 16,   [Dataset: EMNIST_balanced ]                                     : Valid  95 % | Test 88 %    
    
  [now attending]


ver 7.01  + VGG19 5.4_47 : SVM epoch 30, batch8,  kernel='rbf', gamma=0.01, verbose=1, max_iter=100  [Dataset: TrGc_clear_seperate]                                     Test  9%  | TeBc : %
ver 7.02  + VGG19 5.4_47 : SVM epoch 30, batch8,  kernel='rbf', verbose=1, max_iter=1000   [Dataset: TrGc_clear_TrGc_clear_skeletonize]                                : Valid  % | Test   % 


ver 8.01  + VGG19 5.4_47 : SVM epoch 30, batch8,  kernel='rbf', gamma=0.01, verbose=1, max_iter=100  [Dataset: TrGc_clear_seperate_old]                                     Test  9%  | TeBc : %
ver 8.02  + VGG19 5.4_47 : SVM epoch 30, batch8,  kernel='rbf', verbose=1, max_iter=1000             [Dataset: TrGc_clear_skeletonize]                                : Valid  % | Test   % 


[Final model] 

[VGG19]
ver 5.3 VGG19 , batch 16, epoch 10 resize 224, with EMNIST balance Dataset: EMNIST balanced]                        : Valid  90% | test  88% 
ver 4.5 batch 4 epoch 100 resize 224, fc3, with #resize 784 /784, [Dataset: EMNIST byclass]                         : Valid  77% | test  76%   

* Need rename with model.
convnet -> connNet, version start with 6.x

Old
ConnNet_v1.2+VGG19_v5.3 
->
New
ConnNet_v6.x_+_OCR_v5.x_ep00_batch00_



===============================================================================================
'''

epoch_count = 5
version = "8.04"
batch = 8

#   ver1 ~ 3 (26+10)
#   ver4 61 = (26 +26 +10)
#   ver4 47 = 26+10 + 11

# data_dir = '/home/mll/v_mll3/OCR_data/인식_100데이터셋/single_character_Data (사본)/beta_skeletonize'
# data_dir = '/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/dataset/after_skeletonize'
#data_dir = '/home/mll/v_mll3/OCR_data/dataset/MNIST_dataset/EMNIST_balanced'  # emnist_balanced
#data_dir = '/home/mll/v_mll3/OCR_data/dataset/MNIST_dataset/EMNIST_byclass'  # EMNIST_byclass
#ata_dir = "/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/dataset/test_char47"
#data_dir = '/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/dataset/skeletonized data/skeletonized_character_Dataset_1021/'  # skeletonized data
#data_dir = '/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/dataset/seperate_single_character (balance)'  # non -skeletonized
#data_dir = '/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/dataset/generate_img/'
#data_dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TrGc_clear_seperate_old/'

#data_dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc_clear/'
#test_dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/test_sample/'

#data_dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TrGc_clear_seperate_skeletonize/'
#data_dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TrGc_clear_seperate_skeletonize/'
#data_dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc_clear_Deep'
data_dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc_clear_Deep/SK_thin'

TRAIN = 'Train'
VAL = 'Validation'
TEST = 'Test'
save_path = "/home/mll/v_mll3/OCR_data/VGG_character/model/"
log_path = '/home/mll/v_mll3/OCR_data/VGG_character/Log/'
print(torch.cuda.is_available())



#======================================================================================================


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

print("Choose  VGG19 - OCR network model |  1: vgg19_1[47] | 2: vgg19_2[62]")
label=0

model_choose = input()
if model_choose == "1":
    label = 47
    VGG19_47.label = label
    net = VGG19_47.Net()
    net = net.to(device)
    param = list(net.parameters())
    print(len(param))
    for i in param:
        print(i.shape)
    model_type = "VGG19_47"

elif model_choose=="2":
    label = 62
    VGG19_62.label = label
    net = VGG19_62.Net()
    net = net.to(device)
    param = list(net.parameters())
    print(len(param))
    for i in param:
        print(i.shape)
    model_type = "VGG19_62"


print("Choose  Connected Network model |  1: SimpleNet | 2: DTNet | 3: Autoencoder | 4: SVM ")
model_choose = input()
if model_choose == "1":
    ConnNet_Linear.label = label
    net2 = ConnNet_Linear.Net_convol()
    net2 = net2.to(device)
    param = list(net2.parameters())
    print(len(param))
    for i in param:
      print(i.shape)
    model_type2 = "ConnNet_Linear"


elif model_choose=="2":
    ConnNet_DT.label = label
    net2 = ConnNet_DT.Net_DT()
    net2 = net2.to(device)
    param = list(net2.parameters())
    print(len(param))
    for i in param:
        print(i.shape)
    model_type2 = "ConnNet_DT"


elif model_choose=="3":
    ConnNet_Autoencoder.label = label
    net2 = ConnNet_Autoencoder.Net_Autencoder()
    net2 = net2.to(device)
    param = list(net2.parameters())
    print(len(param))
    for i in param:
        print(i.shape)
    model_type2 = "ConnNet_AutoEncoder"

elif model_choose=="4":
    ConnNet_SVM.label = label
    net2 = ConnNet_SVM.Net_SVM()
    net2 = net2.to(device)
    param = list(net2.parameters())
    print(len(param))
    for i in param:
        print(i.shape)
    model_type2 = "ConnNet_svm"



def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def model_loader():
    model_folder = '/home/mll/v_mll3/OCR_data/VGG_character/model'
    model_list = os.listdir(model_folder)
    model_list.sort()
    print("---------------Select model ------------")
    for i in range(0, len(model_list)):
        print("{} : {}".format(i, model_list[i]))
    num = input()
    model_dir = model_folder + '/' + model_list[int(num)]
    model_name = model_list[int(num)]
    # print(model_dir)
    return model_dir, model_name


def train():
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.00001)  # origin  lr =0.00001
    train_size = dataset_sizes[TRAIN]

    for epoch in range(epoch_count):  # loop over the dataset multiple times   #100 epoch -> 3
        running_loss = 0.0
        for i, data in enumerate(dataloaders[TRAIN], 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # print(inputs.shape)
            # print(inputs.shape)
            # forward + backward + optimize

            # outputs, f = net(inputs)               # *original

            outputs, f, result, vector = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (loss.item() > 1000):
                print(loss.item())
                for param in net.parameters():
                    print(param.data)
            # print statistics
            running_loss += loss.item()
            # print(loss.item())
            if i % train_size / 1000 == train_size / (1000 - 1):  # print every 2000 mini-batches
                print('===== [%d, %5d] loss: %.3f ======' %
                      (epoch + 1, i + 1, running_loss))
                # running_loss = 0.0
            if i % 50 == 49:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

        # middle save
        # save_path = "/home/mll/v_mll3/OCR_data/VGG_character/model/skddn_temp_ep{}_ver{}.pth".format(epoch,version)
        # torch.save(net.state_dict(), save_path)
    ''' original
            running_loss += loss.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
    '''

    print('Finished Training')

    save_path = "/home/mll/v_mll3/OCR_data/VGG_character/model/{}_{}_.pth".format(model_type, model_name)
    torch.save(net.state_dict(), save_path)


def validation():
    class_correct = list(0. for i in range(label_count))
    class_total = list(0. for i in range(label_count))
    #class_vector = list([] for i in range(label_count))
    #class_result = list([] for i in range(label_count))

    total_acc = 0
    with torch.no_grad():
        for data in dataloaders[VAL]:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs, _, result, vector = net(images)
            # outputs, _ = net(images)                   #origin
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                #class_vector[label].append(vector)
                #class_result[label].append(result)

    # log file save
    file = open('{}/Validation_log_{}_.txt'.format(log_path, model_name),
                'w')
    file.write("Test dataset dir : {}\n".format(data_dir))
    # vector, result save path.
    newPath = '{}/Valid_log_vector,result_{}'.format(log_path, model_name)
    if not (os.path.isdir(newPath)):  # 새  파일들을 저장할 디렉토리를 생성
        os.makedirs(os.path.join(newPath))

    file.write("Label                               correct count  |  total count \n")
    total_count = 0
    correct_count = 0
    for i in range(label_count):
        print('Accuracy class_correctof %5s : %2d %%' % (
            image_datasets[VAL].classes[i], 100 * class_correct[i] / class_total[i]))
        total_count = total_count + class_total[i]
        correct_count = correct_count + class_correct[i]
        file.write('Accuracy class_correctof %5s : %2d %%' % (
            image_datasets[VAL].classes[i], 100 * class_correct[i] / class_total[i]))
        file.write("    {0:<10}".format(class_correct[i]))
        file.write("| {0:<10} \n".format(class_total[i]))
        '''
        # vector log file save
        file2 = open('{}/Valid_vector_label[{}]_.txt'.format(newPath, image_datasets[VAL].classes[i]), 'w')
        torch.save(vector, "{}/Valid_vector_label[{}]_.pt".format(newPath, image_datasets[VAL].classes[i]))
        file3 = open(
            '{}/Valid_result_label[{}]_.txt'.format(newPath, image_datasets[VAL].classes[i], version, epoch_count), 'w')
        vector2 = vector.tolist()

        # result log file save

        for j in range(len(class_vector[i])):
            file3.write(str(class_result[i][j]))
            file3.write("\n")

        for j in vector2:
            for k in j:
                file2.write(str(k) + " ")
            file2.write("\n\n")

        file2.close()
        file3.close()
        '''
        total_acc = total_acc + (100 * class_correct[i] / class_total[i])

    print('Accuracy total class_correct of  : %2d %%' % (total_acc / label_count))
    file.write('total correct : %2d | total count  %2d  \n' % (correct_count, total_count))
    file.write('Accuracy total class_correct of  : %2d %%' % (total_acc / label_count))

    file.close()
    #file2.close()
    #file3.close()


def test():
    class_correct = list(0. for i in range(label_count))
    class_total = list(0. for i in range(label_count))
    total_acc = 0
    with torch.no_grad():
        for data in dataloaders[TEST]:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs, f, result, vector = net(images)

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    file = open(
        '{} Test_log_{}_.txt'.format(log_path, model_name),
        'w')
    file.write("Test dataset dir : {}\n".format(data_dir))
    file.write("Label                   correct count  |  total count \n")
    total_count = 0
    correct_count = 0
    for i in range(label_count):
        print('Accuracy of %5s : %2d %%' % (
            image_datasets[TEST].classes[i], 100 * class_correct[i] / class_total[i]))
        file.write('Accuracy of %5s : %2d %%' % (
            image_datasets[TEST].classes[i], 100 * class_correct[i] / class_total[i]))
        file.write("    {0:<10}".format(class_correct[i]))
        file.write("| {0:<10} \n".format(class_total[i]))
        total_acc = total_acc + (100 * class_correct[i] / class_total[i])
        total_count = total_count + class_total[i]
        correct_count = correct_count + class_correct[i]

    print('Accuracy total class : %2d %%' % (total_acc / label_count))
    file.write('Accuracy total class : %2d %% \n' % (total_acc / label_count))
    file.write('total correct : %2d | total count  %2d  \n' % (correct_count, total_count))
    file.close()



def train2():

    net.to(device)
    net2.to(device)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.00001)  # origin  lr =0.00001
    train_size = dataset_sizes[TRAIN]




    for epoch in range(epoch_count):  # loop over the dataset multiple times   #100 epoch -> 3
        running_loss = 0.0
        for i, data in enumerate(dataloaders[TRAIN], 0):
            # get the inputs
            inputs, labels = data
            original = labels
            inputs, labels = inputs.to(device), labels.to(device)
            #print(inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()


            # print(inputs.shape)
            # forward + backward + optimize

            # outputs, f = net(inputs)               # *original

            outputs, _, _, vector = net(inputs)
            _, predicted = torch.max(outputs, 1)
            outputs2 = net2(vector)
            _, predicted2 = torch.max(outputs2, 1)
            #print(predicted2, " ", labels)

            loss = criterion(outputs2, labels)
            loss.backward()
            optimizer.step()

            if (loss.item() > 1000):
                print(loss.item())
                for param in net2.parameters():
                    print(param.data)
            # print statistics
            running_loss += loss.item()
            # print(loss.item())
            if i % train_size / 1000 == train_size / (1000 - 1):  # print every 2000 mini-batches
                print('===== [%d, %5d] loss: %.3f ======' %
                      (epoch + 1, i + 1, running_loss))
                # running_loss = 0.0
            if i % 50 == 49:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

        # middle save
        # save_path = "/home/mll/v_mll3/OCR_data/VGG_character/model/skddn_temp_ep{}_ver{}.pth".format(epoch,version)
        # torch.save(net.state_dict(), save_path)
    ''' original
            running_loss += loss.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
    '''

    print('Finished Training')

    save_path = "/home/mll/v_mll3/OCR_data/VGG_character/model/{}_.pth".format(model_name2)
    print(model_name2)
    torch.save(net2.state_dict(), save_path)


def validation2():
    print("valid model:", model_name2)
    class_correct = list(0. for i in range(label_count))
    class_total = list(0. for i in range(label_count))
    TP, TN, FP, FN =0,0,0,0
    class_result = list([] for i in range(label_count))

    total_acc = 0
    with torch.no_grad():
        for data in dataloaders[VAL]:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs, _, _, vector = net(images)
            _, predicted = torch.max(outputs, 1)
            outputs2 = net2(vector)
            _, predicted2 = torch.max(outputs2, 1)
            c = (predicted2 == labels).squeeze()

            #if predicted2 == labels:
             #   TP+=1


            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1



    # log file save
    file = open('{}/Validation_log_{}_.txt'.format(log_path, model_name2),
                'w')
    file.write("Valid dataset dir : {}\n".format(data_dir))
    file.write("epoch : {}, batch {} \n".format(epoch_count, batch))
    file.write("Name : {} \n".format(model_name2))
    # vector, result save path.
    newPath = '{}/Valid_log_vector,result_{}'.format(log_path, model_name2)
    if not (os.path.isdir(newPath)):  # 새  파일들을 저장할 디렉토리를 생성
        os.makedirs(os.path.join(newPath))

    file.write("Label                               correct count  |  total count \n")
    total_count = 0
    correct_count = 0

    for i in range(label_count):
        print('Accuracy class_correctof %5s : %2d %%' % (
            image_datasets[VAL].classes[i], 100 * class_correct[i] / class_total[i]))
        total_count = total_count + class_total[i]
        correct_count = correct_count + class_correct[i]
        file.write('Accuracy class_correctof %5s : %2d %%' % (
            image_datasets[VAL].classes[i], 100 * class_correct[i] / class_total[i]))
        file.write("    {0:<10}".format(class_correct[i]))
        file.write("| {0:<10} \n".format(class_total[i]))

        # result log file save


        total_acc = total_acc + (100 * class_correct[i] / class_total[i])

    print('Accuracy total class_correct of  : %2d %%' % (total_acc / label_count))
    file.write('total correct : %2d | total count  %2d  \n' % (correct_count, total_count))
    file.write('Accuracy total class_correct of  : %2d %%' % (total_acc / label_count))

    file.close()


def test2():
    print("test2 mocel :",model_name2)
    class_correct = list(0. for i in range(label_count))
    class_total = list(0. for i in range(label_count))
    total_acc = 0
    with torch.no_grad():
        for data in dataloaders[TEST]:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs, f, result, vector = net(images)
            _, predicted = torch.max(outputs, 1)
            output2= net2(vector)
            _, predicted2= torch.max(output2, 1)

            c = (predicted2 == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    file = open('{} Test_log_{}_.txt'.format(log_path, model_name2),'w')
    file.write("Test dataset dir : {}\n".format(data_dir))
    file.write("epoch : {}, batch {} \n".format(epoch_count, batch))
    file.write("Name : {} \n".format(model_name2))
    file.write("Label                   correct count  |  total count \n")
    total_count = 0
    correct_count = 0
    for i in range(label_count):
        print('Accuracy of %5s : %2d %%' % (
            image_datasets[TEST].classes[i], 100 * class_correct[i] / class_total[i]))
        file.write('Accuracy of %5s : %2d %%' % (
            image_datasets[TEST].classes[i], 100 * class_correct[i] / class_total[i]))
        file.write("    {0:<10}".format(class_correct[i]))
        file.write("| {0:<10} \n".format(class_total[i]))
        total_acc = total_acc + (100 * class_correct[i] / class_total[i])
        total_count = total_count + class_total[i]
        correct_count = correct_count + class_correct[i]

    print('Accuracy total class : %2d %%' % (total_acc / label_count))
    file.write('Accuracy total class : %2d %% \n' % (total_acc / label_count))
    file.write('total correct : %2d | total count  %2d  \n' % (correct_count, total_count))
    file.close()




#================================================================================================#

#######
# k -fold  or Train val test split
#

data_transforms = {
    TRAIN: transforms.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we randomly crop the image to 224x224 and
        # randomly flip it horizontally.
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    VAL: transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    TEST: transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
}

vector_transform = {
    transforms.Compose([

        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
}

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        transform=data_transforms[x]
    )
    for x in [TRAIN, VAL, TEST]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch,  # origin ver batch= 4 | ver1 batch=2 | ver2  batch=8  | ver batch=4
        shuffle=True, num_workers=4
    )
    for x in [TRAIN, VAL, TEST]
}

dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}
data_shape = image_datasets[TRAIN]

for x in [TRAIN, VAL, TEST]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))

print("Classes: ")
class_names = image_datasets[TRAIN].classes
label_count = len(image_datasets[TRAIN].classes)
print(image_datasets[TRAIN].classes)


model_name = model_type+"_v" + version +"_"+str(label)+"_ep" + str(epoch_count) + "_batch" + str(batch)
print("")
print(model_type)
print(model_type2)
print(data_dir)
print("")
s = 't'
while (s != "1" or s != "2" or s != "3"):
    print("Please command \n [ ( 1 ) train the model |  ( 2 ) load pre-trained model | (3) test_sample_case | \n "
          "(4) train connNet with skeletonized_vector data     |    (5) Load sklearn's SVM   | (6) Train sklearn's  Decision Tree    "
          "(7) Train sklearn's SVM   | (8) sample sklearn's SVM | (9) sklearn's SVM TeBc| (10) sklearn's DT TeBc ] ")
    s = input()
    if s == "1":
        model_name = model_type+"_v" + version +"_"+str(label)+"_ep" + str(epoch_count) + "_batch" + str(batch)
        print(model_name)
        train()
        print("-----------------------")
        validation()
        print("-----------------------")
        test()
        break

    elif s == "2":

        model_dir, model_name = model_loader()
        net.load_state_dict(torch.load(model_dir))
        net.to(device)

        param = list(net.parameters())
        print(len(param))
        for i in param:
            print(i.shape)
        print("-----------------------")
        validation()
        print("-----------------------")
        test()

        break


    elif s == "3":

        model_dir, model_name = model_loader()
        net.load_state_dict(torch.load(model_dir))
        net.to(device)

        param = list(net.parameters())
        print(len(param))
        for i in param:
            print(i.shape)
        print("-----------------------")

        sample_dir = '/home/mll/v_mll3/OCR_data/dataset/MNIST_dataset/sample/'
        sample_list = os.listdir(sample_dir)



        for i in sample_list:
            img = Image.open("{}/{}".format(sample_dir, i))
            if (img.mode == "L"):
                img = img.convert("RGB")

            img_label = i.split("_")[1][0]
            predicted = 0
            img = data_transforms[TEST](img)
            img = img.unsqueeze(0)
            image = img.cuda()
            net.eval()
            outputs, f, result, vector = net(image)
            _, predicted = torch.max(outputs, 1)
            print("file {} :  label : {} | predict: {}".format(i, img_label, class_names[predicted]))

        break

    elif s == "4":

        print("Select VGG19 model :")
        model_dir, model_name = model_loader()
        net.load_state_dict(torch.load(model_dir))

        net.to(device)
        vgg19_v = model_name.split("_")[1]

        model_name2 = model_type2+"_"+str(label)+"_v" + version + "+" + vgg19_v
        print("Train model : ", model_name2)

        train2()
        print("-----------------------")
        validation2()
        print("-----------------------")
        test2()
        break

    elif s == "5":


        print("Select VGG19 model to valid :")
        model_dir, model_name = model_loader()
        net.load_state_dict(torch.load(model_dir))
        net.to(device)

        print("Select ConnNet model to valid :")
        model_dir2, model_name2= model_loader()
        svm = joblib.load(model_dir2)

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        print(model_name+" + "+ model_name2 +" : target model.")
        print("test.." + str(dataset_sizes[TEST]))
        for data in dataloaders[TEST]:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs, f, result, vector = net(images)
            _, predicted = torch.max(outputs, 1)

            vector2 = vector.to(torch.device("cpu"))
            labels2 = labels.to(torch.device("cpu"))

            np_vector = vector2.detach().numpy()
            np_labels = labels2.detach().numpy()
            for j in range(0, len(np_vector)):
                x_test.append(np_vector[j])
                y_test.append(np_labels[j])
            y_pred = svm.predict(x_test)

        file = open('{}/Valid_SVM_{}_log_{}.txt'.format(log_path, model_name2), 'w')
        file.write("Test dataset dir : {}\n".format(data_dir))
        file.write('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
        file.write('Accuracy: %.2f' % svm.score(x_test, y_pred))
        file.write('F1 score : %.2f' % f1_score(y_test, y_pred, average='micro'))
        file.close()


        print("{} saved.".format(model_name2))
        print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
        print('F1 score : %.2f' % f1_score(y_test, y_pred, average='micro'))

        print("-----------------------")

        break

    elif s == "6":      # Decision tree

        print("Select VGG19 model :")
        model_dir, model_name = model_loader()
        net.load_state_dict(torch.load(model_dir))
        net.to(device)

        model_type2="ConnNet_DT"
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        i = 0
        vgg19_v = model_name.split("_")[1]
        model_name2 = model_type2 + "_" + str(label) + "_v" + version + "+" + vgg19_v
        print("Train model : ", model_name2)

        print("train..")
        #x_train = np.load("/home/mll/v_mll3/OCR_data/VGG_character/np/{}_x_train.npy".format(model_name2))
        #y_train = np.load("/home/mll/v_mll3/OCR_data/VGG_character/np/{}_y_train.npy".format(model_name2))
        #if x_train.size == 0:
        print("NO np.. loading...")
        for data in dataloaders[TRAIN]:

            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs, f, result, vector = net(images)
            _, predicted = torch.max(outputs, 1)

            vector2 = vector.to(torch.device("cpu"))
            labels2 = labels.to(torch.device("cpu"))

            np_vector = vector2.detach().numpy()
            np_labels = labels2.detach().numpy()
            # print(len(np_vector))
            for j in range(0, len(np_vector)):
                x_train.append(np_vector[j])
                y_train.append(np_labels[j])

        np.save('/home/mll/v_mll3/OCR_data/VGG_character/np/{}_x_train'.format(model_name2), x_train)
        np.save('/home/mll/v_mll3/OCR_data/VGG_character/np/{}_y_train'.format(model_name2), y_train)

        print("fit..")
        dt_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0)
        dt_tree.fit(x_train, y_train)


        print("test..")
        x_test =np.load("/home/mll/v_mll3/OCR_data/VGG_character/np/{}_x_test.npy".format(model_name2))
        y_test =np.load("/home/mll/v_mll3/OCR_data/VGG_character/np/{}_y_test.npy".format(model_name2))
        #if x_test.size == 0:
        '''
        print("NO np.. loading...")
        for data in dataloaders[TEST]:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs, f, result, vector = net(images)
            _, predicted = torch.max(outputs, 1)

            vector2 = vector.to(torch.device("cpu"))
            labels2 = labels.to(torch.device("cpu"))

            np_vector = vector2.detach().numpy()
            np_labels = labels2.detach().numpy()
            for j in range(0, len(np_vector)):
                x_test.append(np_vector[j])
                y_test.append(np_labels[j])
        np.save('/home/mll/v_mll3/OCR_data/VGG_character/np/{}_x_test'.format(model_name2), x_test)
        np.save('/home/mll/v_mll3/OCR_data/VGG_character/np/{}_y_test'.format(model_name2), y_test)
        '''
        print("DT..saving...")
        joblib.dump(dt_tree, '/home/mll/v_mll3/OCR_data/VGG_character/model/{}.pkl'.format(model_name2))
        y_pred = dt_tree.predict(x_test)
        print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

        print("-----------------------")
        file = open('{}/Log_{}.txt'.format(log_path,model_name2), 'w')
        file.write("Test dataset dir : {}\n".format(data_dir))
        file.write("Test acc : {}\n".format(accuracy_score(y_test, y_pred)))
        file.write(classification_report(y_test, y_pred, target_names=class_names, digits=4))

        break

    elif s == "7":  # SVM train

        print("Select VGG19 model :")
        model_dir, model_name = model_loader()
        net.load_state_dict(torch.load(model_dir))
        net.to(device)

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        i = 0

        model_type2 ="ConnNet_SVM"
        vgg19_v = model_name.split("_")[1]
        model_name2 = model_type2 + "_" + str(label) + "_v" + version + "+" + vgg19_v
        print("Train model : ", model_name2)

        #temp ="ConnNet_Linear_47_v7.01+Ver5.4"
        print("train..")
        x_train = np.load("/home/mll/v_mll3/OCR_data/VGG_character/np/ConnNet_DT_47_v8.02+Ver5.4_x_train.npy")
        y_train = np.load("/home/mll/v_mll3/OCR_data/VGG_character/np/ConnNet_DT_47_v8.02+Ver5.4_y_train.npy")
        '''
        if x_train.size == 0 :
            print("NO np.. loading...")
            for data in dataloaders[TRAIN]:

                images, labels = data
                images = images.cuda()
                labels = labels.cuda()

                outputs, f, result, vector = net(images)
                _, predicted = torch.max(outputs, 1)

                vector2 = vector.to(torch.device("cpu"))
                labels2 = labels.to(torch.device("cpu"))

                np_vector = vector2.detach().numpy()
                np_labels = labels2.detach().numpy()
                # print(len(np_vector))
                for j in range(0, len(np_vector)):
                    x_train.append(np_vector[j])
                    y_train.append(np_labels[j])

            np.save('/home/mll/v_mll3/OCR_data/VGG_character/np/{}_x_train'.format(model_name2),x_train)
            np.save('/home/mll/v_mll3/OCR_data/VGG_character/np/{}_y_train'.format(model_name2),y_train)
        '''
        print("fit train data...")
        vector_svm = svm.SVC(kernel='rbf', gamma=0.1, verbose=1, max_iter=100)  #max_iter =-1 (no limit)
        kernal = 'rbf'
        #kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        vector_svm.fit(x_train, y_train).score(x_train, y_train)

        x_test = np.load("/home/mll/v_mll3/OCR_data/VGG_character/np/ConnNet_DT_47_v8.02+Ver5.4_x_test.npy")
        y_test = np.load("/home/mll/v_mll3/OCR_data/VGG_character/np/ConnNet_DT_47_v8.02+Ver5.4_y_test.npy")
        print("test..")
        '''
        if x_train.size == 0 :
            print("NO np.. loading...")
            for data in dataloaders[TEST]:
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()

                outputs, f, result, vector = net(images)
                _, predicted = torch.max(outputs, 1)

                vector2 = vector.to(torch.device("cpu"))
                labels2 = labels.to(torch.device("cpu"))

                np_vector = vector2.detach().numpy()
                np_labels = labels2.detach().numpy()
                for j in range(0, len(np_vector)):
                    x_test.append(np_vector[j])
                    y_test.append(np_labels[j])
            np.save('/home/mll/v_mll3/OCR_data/VGG_character/np/{}_x_test'.format(model_name2),x_test)
            np.save('/home/mll/v_mll3/OCR_data/VGG_character/np/{}_y_test'.format(model_name2),y_test)
        '''
        y_pred = vector_svm.predict(x_test)
        #average_precision = average_precision_score(y_test, y_pred)


        joblib.dump(vector_svm, '/home/mll/v_mll3/OCR_data/VGG_character/model/{}.pkl'.format(model_name2))
        print("{} saved.".format(model_name2))
        print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
        print('F1 score : %.4f' % f1_score(y_test, y_pred, average='micro'))
        print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

        print("-----------------------")
        file = open('{}/SVM_{}_log_{}.txt'.format(log_path,kernal, model_name2), 'w')
        file.write("Test dataset dir : {}\n".format(data_dir))
        file.write('Accuracy: %.4f\n' % accuracy_score(y_test, y_pred))
        file.write('F1 score : %.4f\n' % f1_score(y_test, y_pred, average='micro'))
        file.write(classification_report(y_test, y_pred, target_names=class_names, digits=4))
        file.close()
        break

    elif s == "8":  # SVM sample test

        print("Select VGG19 model to valid :")
        model_dir, model_name = model_loader()
        net.load_state_dict(torch.load(model_dir))
        net.to(device)

        print("Select ConnNet model to valid :")
        model_dir2, model_name2 = model_loader()
        svm = joblib.load(model_dir2)

        print(model_name + " + " + model_name2 + " : target model.")

        sample_dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc/IC13'
        sample_list = os.listdir(sample_dir)

        for i in sample_list:
            img = Image.open("{}/{}".format(sample_dir, i))
            if (img.mode == "L"):
                img = img.convert("RGB")

            img_label = i
            predicted = 0
            img = data_transforms[TEST](img)
            img = img.unsqueeze(0)
            image = img.cuda()
            #net.eval()
            outputs, f, result, vector = net(image)
            _, predicted = torch.max(outputs, 1)

            vector2 = vector.to(torch.device("cpu"))
            labels2 = img_label

            np_vector = vector2.detach().numpy()
            #np_labels = labels2.numpy()

            y_pred = svm.predict(np_vector)
            print("file {} :  label : {} | ocr predict : {} | y_pred : {} -> {}".format(i, img_label, class_names[predicted], y_pred, class_names[y_pred[0]]))
            #print(class_names)




        print("-----------------------")

        break

    elif s == "9":  # SVM  test TeBc

        print("Select VGG19 model to valid :")
        model_dir, model_name = model_loader()
        net.load_state_dict(torch.load(model_dir))
        net.to(device)

        print("Select ConnNet model to valid :")
        model_dir2, model_name2 = model_loader()
        svm = joblib.load(model_dir2)

        print(model_name + " + " + model_name2 + " : target model.")

        x_test = []
        y_test = []

        print("test.."Unreeeeunununuununun


        )

        x_test = np.load("/home/mll/v_mll3/OCR_data/VGG_character/np/ConnNet_DT_47_v8.02+Ver5.4.pkl_x_test.npy")
        y_test = np.load("/home/mll/v_mll3/OCR_data/VGG_character/np/ConnNet_DT_47_v8.02+Ver5.4.pkl_y_test.npy")

        y_pred = svm.predict(x_test)
        # average_precision = average_precision_score(y_test, y_pred)


        #print("{} saved.".format(model_name2))
        print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
        print('F1 score : %.4f' % f1_score(y_test, y_pred, average='micro'))
        print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
        print("-----------------------")


    elif s == "10":  # DT  test TeBc

        print("Select VGG19 model to valid :")
        model_dir, model_name = model_loader()
        net.load_state_dict(torch.load(model_dir))
        net.to(device)

        print("Select ConnNet model to valid :")
        model_dir2, model_name2 = model_loader()
        dt_tree = joblib.load(model_dir2)

        print(model_name + " + " + model_name2 + " : target model.")

        x_test = []
        y_test = []

        print("test..")
        print("NO np.. loading...")
        for data in dataloaders[TEST]:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs, f, result, vector = net(images)
            _, predicted = torch.max(outputs, 1)

            vector2 = vector.to(torch.device("cpu"))
            labels2 = labels.to(torch.device("cpu"))

            np_vector = vector2.detach().numpy()
            np_labels = labels2.detach().numpy()
            for j in range(0, len(np_vector)):
                x_test.append(np_vector[j])
                y_test.append(np_labels[j])
        np.save('/home/mll/v_mll3/OCR_data/VGG_character/np/{}_x_test'.format(model_name2), x_test)
        np.save('/home/mll/v_mll3/OCR_data/VGG_character/np/{}_y_test'.format(model_name2), y_test)

       # x_test = np.load("/home/mll/v_mll3/OCR_data/VGG_character/np/TeBc_clear_x_test.npy")   #Sk_thin_TeBc_clear
       # y_test = np.load("/home/mll/v_mll3/OCR_data/VGG_character/np/TeBc_clear_y_test.npy")

        print("predict..")
        y_pred = dt_tree.predict(x_test)
        # average_precision = average_precision_score(y_test, y_pred)

        print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
        print('F1 score : %.4f' % f1_score(y_test, y_pred, average='micro'))
        print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
        print("-----------------------")

    elif s == "11":  # DT  sample

        print("Select VGG19 model to valid :")
        model_dir, model_name = model_loader()
        net.load_state_dict(torch.load(model_dir))
        net.to(device)

        print("Select ConnNet model to valid :")
        model_dir2, model_name2 = model_loader()
        dt_tree = joblib.load(model_dir2)

        print(model_name + " + " + model_name2 + " : target model.")

        log_file = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc/DT_detect_not_deep.txt','w')

        sample_dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc_notDeep/'
        label_file = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc_clear_Deep/save_log.txt', 'r')
        sample_list = os.listdir(sample_dir)
        label_dict = {}

        for i in label_file:
            dir = i.split("\t")[0].replace("/","_")
            label = i.split("\t")[2].strip()

            label_dict.update({dir : label})

        count =0
        label47 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'E', 'F', 'G', 'H', 'N', 'Q', 'R',
                   'T', 'a', 'b',
                   'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                   'w', 'x', 'y', 'z']
        import random



        sample_list.sort()
        for i in sample_list:
            img = Image.open("{}/{}".format(sample_dir, i))
            #if (img.mode == "L"):
            img = img.convert("RGB")

            img_label = label_dict[i]
            predicted = 0
            img = data_transforms[TEST](img)
            img = img.unsqueeze(0)
            image = img.cuda()
            #net.eval()
            outputs, f, result, vector = net(image)
            _, predicted = torch.max(outputs, 1)

            vector2 = vector.to(torch.device("cpu"))
            labels2 = img_label

            np_vector = vector2.detach().numpy()
            #np_labels = labels2.numpy()

            y_pred = dt_tree.predict(np_vector)
            #random_label = label47[random.randint(0, 46)]
            print("{} :  label : {} | ocr predict : {} | y_pred:{}\t{}\t{}\n".format(i, img_label, class_names[predicted], y_pred, class_names[y_pred[0]], img_label==class_names[y_pred[0]]))
            log_file.write("{} :  label : {} | ocr predict : {} | y_pred:{}\t{}\t{}\n".format(i, img_label, class_names[predicted], y_pred, class_names[y_pred[0]], img_label== class_names[y_pred[0]]))
            if  img_label==class_names[y_pred[0]]:
                count+=1
        print(len(sample_list))
        print(count)


        print("-----------------------")
    else:
        print("check command")

print("-----------------------")

print("Done!")
