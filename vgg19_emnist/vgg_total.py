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
from sklearn import tree
from sklearn import datasets as sk_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score


'''
===============================================================================================
[Based network]
 Pytorch VGG19

[Dataset] 
v1.x English single character set(external data)

v2.x skeletonized external data

v3.x Skeletonized_character_dataset6  : Recognition character ->  dataset

v4.x EMNIST OCR 62 label dataset *

ver 6.0x Simple network with original skeletonized dataset (crop from CRAFT, Deep-text data)
ver 6.1x Simple network with generated synth char dataset 

ver 7.0x Decision Tree

-----------------------------------------------------
2020.09.28 mon

Data rebuliding...
version sequence also change..



[Lastest update] : 2020.11.10

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
ver 4.6 batch 4 epoch 10 resize 224 balanced   [now training]
ver 4.7 batch 4 epoch 30 resize 224 balanced   

ver 5.3 VGG19 , batch 16, epoch 10 resize 224, Data : EMNIST balance
ver 5.4 VGG19,  batch 16, epoch 20, resize 224, label 47,                       [EMNIST_balance : valid 94% | Test 88%]

[EMNIST_Letter_vgg and spinalVGG.py]xxxxxxxxxxxxxx
ver 5.0 spinalnet + vgg5 with EMNIST byclass    
ver 5.1 spinalnet + vgg5 with EMNIST balance     Valid : 90 | test : 24 


================[ConnNet]======================

ver 6.x Simple network
ver 6.1x = 47 label
ver 6.xx = 62 label

ver 6.11 + VGG19 5.4_47 : simplenet epoch 30, batch16 [Dataset: skeletonized_character_Dataset_1021] : Valid 2% | Test 2% 
ver 6.12 + VGG19 5.4_47 : simplenet epoch 20, batch16 [Dataset: skeletonized_character_Dataset_1021] : Valid 2% | Test 1% 
ver 6.13 + VGG19 5.4_47 : simplenet epoch 100, batch16, net_conv2 [Dataset: skeletonized_character_Dataset_1021] : Valid  2% | Test   1%
ver 6.14 + VGG19 5.4_47 : autoencoder epoch 30, batch16, net_conv2 [Dataset: skeletonized_character_Dataset_1021] : Valid  2% | Test   1%
ver 6.15 + VGG19 5.4_47 : autoencoder epoch 10, batch16, net_conv2 [Dataset: skeletonized_character_Dataset_1021] : Valid  2% | Test   2%
ver 6.16 + VGG19 5.4_47 : simplenet epoch 10, batch16, net_conv2 [Dataset: test_char47] : Valid  2% | Test   1%


ver 7.x Decision Tree




[Final model] 

[VGG19]
ver 5.3 VGG19 , batch 16, epoch 10 resize 224, with EMNIST balance      [Dataset: balanced] : Valid : 90% | test : 88% : 
ver 4.5 batch 4 epoch 100 resize 224, fc3, with EMNIST byclass          [Dataset: byclass] Valid : 77% | test : 76%     # resize 784 /784

* Need rename with model.
convnet -> connNet, version start with 6.x

Old
ConnNet_v1.2+VGG19_v5.3 
->
New
ConnNet_v6.x_+_OCR_v5.x_ep00_batch00_



===============================================================================================
'''

epoch_count = 10
version = "6.16"
batch = 16

#   ver1 ~ 3 (26+10)
#   ver4 61 = (26 +26 +10)
#   ver4 47 = 26+10 + 11

# data_dir = '/home/mll/v_mll3/OCR_data/인식_100데이터셋/single_character_Data (사본)/beta_skeletonize'
# data_dir = '/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/dataset/after_skeletonize'
#data_dir = '/home/mll/v_mll3/OCR_data/dataset/MNIST_dataset/EMNIST_balanced'  # emnist_balanced
data_dir = '/home/mll/v_mll3/OCR_data/dataset/MNIST_dataset/EMNIST_byclass'  # EMNIST_byclass
#data_dir = "/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/dataset/test_char47"
#data_dir = '/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/dataset/skeletonized_character_Dataset_1021'  # skeletonized data

TRAIN = 'Train'
VAL = 'Validation'
TEST = 'Test'
save_path = "/home/mll/v_mll3/OCR_data/VGG_character/model/"
log_path = '/home/mll/v_mll3/OCR_data/VGG_character/Log/'
print(torch.cuda.is_available())
'''
transform = transforms.Compose([
    transforms.Resize(784),         #transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
'''


# ---------------------------------------
# VGG-19_62 label, additional 3 nn layer
class Net2(nn.Module):
    label = 62

    def __init__(self):
        super(Net2, self).__init__()
        self.conv = nn.Sequential(
            # 3 224 128
            nn.Conv2d(3, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 64 112 64
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 128 56 32
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 256 28 16
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 512 14 8
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )
        # 512 7 4

        self.avg_pool = nn.AvgPool2d(7)
        # 512 1 1

        # self.classifier = nn.Linear(512, label)

        self.fc1 = nn.Linear(512 * 2 * 2, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 512)

        self.classifier = nn.Linear(512, label)

    def forward(self, x):
        # print(x.size())
        features = self.conv(x)
        # print(features.size())
        x = self.avg_pool(features)
        x = x.view(features.size(0), -1)
        vector = x
        x = self.classifier(x)
        result = x

        # x = self.softmax(x)
        # result = self.softmax(x)
        # vector = self.last_vector

        return x, features, result, vector

# VGG-19_47 label, normal vgg19
class Net(nn.Module):
    label = 47

    # vector set of classify.

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            # 3 224 128
            nn.Conv2d(3, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 64 112 64
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 128 56 32
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 256 28 16
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 512 14 8
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )
        # 512 7 4

        self.avg_pool = nn.AvgPool2d(7)
        # 512 1 1

        self.classifier = nn.Linear(512, label)
        """
        self.fc1 = nn.Linear(512*2*2,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)


        self.classifier = nn.Linear(512, label)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, label),
        )
        """

    def forward(self, x):
        # print(x.size())
        features = self.conv(x)
        # print(features.size())
        x = self.avg_pool(features)
        x = x.view(features.size(0), -1)
        vector = x
        x = self.classifier(x)
        result = x

        # x = self.softmax(x)
        # result = self.softmax(x)
        # vector = self.last_vector

        return x, features, result, vector

# Connected Network -1 , simple layers
class Net_convol2(nn.Module):
    def __init__(self):
        super(Net_convol2, self).__init__()
        self.fc1 = nn.Linear(512, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 60)
        self.classifier = nn.Linear(60, label)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.classifier(x)
        return x

# Connected Network -1.1 , simple layers, edit some linear parameter size
class Net_convol(nn.Module):
    def __init__(self):
        super(Net_convol, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 100)
        self.classifier = nn.Linear(100, label)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.classifier(x)
        return x

    def _cal_penalty(self, layer_idx, _mu, _path_prob):

        penalty = torch.tensor(0.).to(self.device)

        batch_size = _mu.size()[0]
        _mu = _mu.view(batch_size, 2 ** layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))

        for node in range(0, 2 ** (layer_idx + 1)):
            alpha = (torch.sum(_path_prob[:, node] * _mu[:, node // 2], dim=0) /
                     torch.sum(_mu[:, node // 2], dim=0))

            layer_penalty_coeff = self.penalty_list[layer_idx]

            penalty -= 0.5 * layer_penalty_coeff * (torch.log(alpha) +
                                                    torch.log(1 - alpha))

        return penalty

    """ 
      Add a constant input `1` onto the front of each instance. 
    """

    def _data_augment(self, X):
        batch_size = X.size()[0]
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(self.device)
        X = torch.cat((bias, X), 1)

        return X

    def _validate_parameters(self):

        if not self.depth > 0:
            msg = 'The tree depth should be strictly positive, but got {} instead.'
            raise ValueError(msg.format(self.depth))

        if not self.lamda >= 0:
            msg = ('The coefficient of the regularization term should not be'
                   ' negative, but got {} instead.')
            raise ValueError(msg.format(self.lamda))

# Connected Network -2 Decision Tree        https://github.com/AaronX121/Soft-Decision-Tree/
class Net_DT(nn.Module):

    def __init__(self, input_dim, output_dim, depth=5, lamda=1e-3, use_cuda=False):
        super(Net_DT, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.lamda = lamda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self._validate_parameters()
        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        # Different penalty coefficients for nodes in different layers
        self.penalty_list = [self.lamda * (2 ** (-depth))
                             for depth in range(0, self.depth)]

        # Initialize internal nodes and leaf nodes, the input dimension on
        # internal nodes is added by 1, serving as the bias.
        self.inner_nodes = nn.Sequential(
            nn.Linear(self.input_dim + 1,
                      self.internal_node_num_, bias=False),
            nn.Sigmoid())

        self.leaf_nodes = nn.Linear(self.leaf_node_num_,
                                    self.output_dim, bias=False)

    def forward(self, X, is_training_data=False):

        _mu, _penalty = self._forward(X)
        y_pred = self.leaf_nodes(_mu)

        # When `X` is the training data, the model also returns the penalty
        # to compute the training loss.
        if is_training_data:
            return y_pred, _penalty
        else:
            return y_pred

class Net_Autencoder(nn.Module):
    def __init__(self):
        super(Net_Autencoder, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)
        self.fc3 = nn.Linear(10, 256)
        self.fc4 = nn.Linear(256, 100)
        self.classifier = nn.Linear(100, label)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.classifier(x)
        return x


#==================================================================


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

print("Choose  VGG19 - OCR network model |  1: vgg19_1[47] | 2: vgg19_2[62]")
label=0

model_choose = input()
if model_choose == "1":
    label = 47
    net = Net()
    net = net.to(device)
    param = list(net.parameters())
    print(len(param))
    for i in param:
        print(i.shape)
    model_type = "VGG19_47"

elif model_choose=="2":
    label = 62
    net = Net2()
    net = net.to(device)
    param = list(net.parameters())
    print(len(param))
    for i in param:
        print(i.shape)
    model_type = "VGG19_62"


print("Choose  Connected Network model |  1: SimpleNet | 2: DTNet | 3: Autoencoder")
model_choose = input()
if model_choose == "1":
    net2 = Net_convol()
    net2 = net2.to(device)
    param = list(net2.parameters())
    print(len(param))
    for i in param:
      print(i.shape)
    model_type = "ConnNet_Linear"

elif model_choose=="2":
    net2 = Net_DT()
    net2 = net2.to(device)
    param = list(net2.parameters())
    print(len(param))
    for i in param:
        print(i.shape)
    model_type = "ConnNet_DT"

elif model_choose=="3":
    net2 = Net_Autencoder()
    net2 = net2.to(device)
    param = list(net2.parameters())
    print(len(param))
    for i in param:
        print(i.shape)
    model_type = "ConnNet_AutoEncoder"
# print(param[0].shape)


# functions to show an image


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



def train2(model_name2):

    net2 = Net_convol()
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


def validation2(model_name2):
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

    file = open(
        '{} Test_log_{}_.txt'.format(log_path, model_name2),
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
print(model_name)
print("")
print(data_dir)
print("")
s = 't'
while (s != "1" or s != "2" or s != "3"):
    print("Please command \n [ ( 1 ) train the model |  ( 2 ) load pre-trained model | (3) test_sample_case | \n "
          "(4) train connNet with skeletonized_vector data   (5) validation connNet ] ")
    s = input()
    if s == "1":

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

        model_name2 = model_type+"_"+str(label)+"_v" + version + "+" + vgg19_v
        print("Train model : ", model_name2)

        train2(model_name2)
        print("-----------------------")
        validation2(model_name2)
        print("-----------------------")
        test2()
        break

    elif s == "5":


        print("Select VGG19 model :")
        model_dir, model_name = model_loader()
        net.load_state_dict(torch.load(model_dir))
        net.to(device)

        print("Select ConnNet model :")
        net2 = Net_convol()
        model_dir2, model_name2= model_loader()
        net2.load_state_dict(torch.load(model_dir2))
        net2.to(device)

        print("-----------------------")
        validation2(model_name2)
        print("-----------------------")
        test2()
        break

    else:
        print("check command")

print("-----------------------")

print("Done!")