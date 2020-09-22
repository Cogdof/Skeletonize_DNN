import io

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
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
from torch.utils.data import Dataset, DataLoader
import pandas as pd
'''
==================================================
[Based network]
 Pytorch VGG19

[Dataset] 
v1.x English single character set(external data)

v2.x skeletonized external data

v3.x Skeletonized_character_dataset6  : Recognition character ->  dataset

v4.x Skeletonized_word_to_char_dataset7 : Skeletonize Pass_word_dataset4,  → dataset

-----------------------------------------------------
2020.09.08

Data rebuliding...
version sequence also change..





[Lastest update]
2020.09.22  Tue 

[version]
ver 1.0 batch 8, epoch 5
ver 1.1 batch 8, epoch 10
ver 1.2 batch 8, epoch 20
ver 1.3 batch 4, epoch 5

ver 2.1 batch 8 epoch 5, skeletonize(external data)
ver 2.2 batch 8 epoch 10, skeletonize(external data)
ver 2.3 batch 4, epoch 10, skeletonize(external data)
ver 2.4 batch 4, epoch 5, skeletonize(external data).
    change f1,f2,f3 layers

ver b3.0 batch 4. epoch 5, dataset6 (label : 52 a~z, A~Z, non numberic)

ver 4.0 VGG 26+26+10 case,digit distinguish re train


==================================================
'''

epoch_count = 10
version = "4.0"
batch = 4
label = 35
#   ver1 ~ 3 (26+10)
#   ver4 61 = (26 +26 +10)

model_name = "OCR_vgg_ver" + version + "_ep" + str(epoch_count) + "_batch" + str(batch)

# data_dir = '/home/mll/v_mll3/OCR_data/인식_100데이터셋/single_character_Data (사본)/beta_skeletonize'
# data_dir = '/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/dataset/after_skeletonize'
#data_dir = '/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/dataset/data'
data_dir = '/home/mll/v_mll3/OCR_data/dataset/MNIST_dataset/EMNIST_byclass'
TRAIN = 'Train'
VAL = 'Validation'
TEST = 'Test'
save_path = "/home/mll/v_mll3/OCR_data/VGG_character/model/"

print(torch.cuda.is_available())

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


'''
# data_dir = '/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/dataset/after_skeletonize'
data_dir = '/home/mll/v_mll3/OCR_data/dataset/MNIST_dataset/EMNIST_byclass/'
TRAIN = 'Train'
VAL = 'Validation'
TEST = 'Test'
save_path = "/home/mll/v_mll3/OCR_data/VGG_character/model/"

input_size = 28
batch_size = 64

train_data_dir = data_dir+"emnist-byclass-train.csv"
test_data_dir = data_dir+"emnist-byclass-test.csv"

train_data = pd.read_csv(train_data_dir)
test_data = pd.read_csv(test_data_dir)

print(train_data.shape)
print(test_data.shape)
'''


class Net(nn.Module):

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

        # print(self)
        # last vector
        # self.last_vector = self

        self.classifier = nn.Linear(512, label)
        """
        self.fc1 = nn.Linear(512*2*2,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
        """

    def forward(self, x):
        # print(x.size())
        features = self.conv(x)
        # print(features.size())
        x = self.avg_pool(features)
        # vector = x
        # print(avg_pool.size())
        x = x.view(features.size(0), -1)
        # print(flatten.size())
        vector = x
        x = self.classifier(x)
        result = x

        # x = self.softmax(x)
        # result = self.softmax(x)
        # vector = self.last_vector

        return x, features, result, vector


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = Net()
net = net.to(device)
param = list(net.parameters())
print(len(param))
for i in param:
    print(i.shape)


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


# VGG-16 Takes 224x2


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

            # predicted = torch.max(outputs, 1)

            # print(labels)
            # print(predicted)
            # print(result)
            # print(vector)
            # print(outputs.shape)
            # print(labels.shape)1

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

    save_path = "/home/mll/v_mll3/OCR_data/VGG_character/model/{}_.pth".format(model_name)
    torch.save(net.state_dict(), save_path)


def validation():
    class_correct = list(0. for i in range(label_count))
    class_total = list(0. for i in range(label_count))
    class_vector = list([] for i in range(label_count))
    class_result = list([] for i in range(label_count))

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
                class_vector[label].append(vector)
                class_result[label].append(result)

    # log file save
    file = open('/home/mll/v_mll3/OCR_data/VGG_character/Skeletonize_DNN/Log/Validation_log_{}_.txt'.format(model_name),
                'w')

    # vector, result save path.
    newPath = '/home/mll/v_mll3/OCR_data/VGG_character/Skeletonize_DNN/Log/Valid_log_vector,result_{}'.format(
        model_name)
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
        total_acc = total_acc + (100 * class_correct[i] / class_total[i])

    print('Accuracy total class_correct of  : %2d %%' % (total_acc / label_count))
    file.write('total correct : %2d | total count  %2d  \n' % (correct_count, total_count))
    file.write('Accuracy total class_correct of  : %2d %%' % (total_acc / label_count))

    file.close()
    file2.close()
    file3.close()


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
        '/home/mll/v_mll3/OCR_data/VGG_character/Skeletonize_DNN/Log/Test_log_{}_.txt'.format(model_name),
        'w')
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


# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transforms = {
    TRAIN: transforms.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we randomly crop the image to 224x224 and
        # randomly flip it horizontally.
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
    TEST: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
}


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.file = file_path
        self.data = pd.read_csv(file_path, dtype=np.float32)
        self.label = self.data.label.values
        self.len = len(self.label)
        self.data = self.data.values[:, 1:] / 255

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len


train_csv = '/home/mll/v_mll3/OCR_data/dataset/MNIST_dataset/EMNIST_byclass/Train/emnist-byclass-train.csv'
test_csv = '/home/mll/v_mll3/OCR_data/dataset/MNIST_dataset/EMNIST_byclass/Test/emnist-byclass-test.csv'

train_dataloader = torch.utils.data.DataLoader(
    MNISTDataset(train_csv), batch_size=batch, shuffle=True, num_workers=4
)

test_dataloader = torch.utils.data.DataLoader(
    MNISTDataset(test_csv), batch_size=batch, shuffle=True, num_workers=4
)

#csv_dataset_sizes = {x: len(train_dataloader[x]) for x in [TRAIN,TEST]}


print("Loaded {} images under {}".format(train_dataloader,train_csv))
print("Loaded {} images under {}".format(test_dataloader,test_csv))
print("Classes: ")
class_names = train_dataloader.classes
label_count = len(train_dataloader.classes)
print(train_dataloader.classes)


s = 't'
while (s != "1" or s != "2"):
    print("Please command \n [ ( 1 ) train the model |  ( 2 ) load pre-trained model | (3) test_sample_case ] ")
    s = input()
    if s == "1":

        # need to set epoch, version info.

        train()
        print("-----------------------")
        validation()
        print("-----------------------")
        test()
        break


    elif s == "2":
        net = Net()
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
        net = Net()
        model_dir, model_name = model_loader()
        net.load_state_dict(torch.load(model_dir))
        net.to(device)

        param = list(net.parameters())
        print(len(param))
        for i in param:
            print(i.shape)
        print("-----------------------")
        break;

    else:
        print("check command")

print("-----------------------")
print("Done.")
