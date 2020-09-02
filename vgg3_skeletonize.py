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




print(torch.cuda.is_available())


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            #3 224 128
            nn.Conv2d(3, 64, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #64 112 64
            nn.Conv2d(64, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #128 56 32
            nn.Conv2d(128, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #256 28 16
            nn.Conv2d(256, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #512 14 8
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )
        #512 7 4

        self.avg_pool = nn.AvgPool2d(7)
        #512 1 1
        self.classifier = nn.Linear(512, 35)
        """
        self.fc1 = nn.Linear(512*2*2,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
        """

    def forward(self, x):

        #print(x.size())
        features = self.conv(x)
        #print(features.size())
        x = self.avg_pool(features)
        #print(avg_pool.size())
        x = x.view(features.size(0), -1)
        #print(flatten.size())
        x = self.classifier(x)
        #x = self.softmax(x)
        return x, features

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = Net()
net = net.to(device)
param = list(net.parameters())
print(len(param))
for i in param:
    print(i.shape)
#print(param[0].shape)


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


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
            outputs, f = net(inputs)
            # print(outputs.shape)
            # print(labels.shape)
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
                #running_loss = 0.0
            if i % 50 == 49:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

        # middle save
        #save_path = "/home/mll/v_mll3/OCR_data/VGG_character/model/skddn_temp_ep{}_ver{}.pth".format(epoch,version)
        #torch.save(net.state_dict(), save_path)
    ''' original
            running_loss += loss.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
    '''

    print('Finished Training')

    save_path = "/home/mll/v_mll3/OCR_data/VGG_character/model/skdnn_vgg_{}_ver{}.pth".format(epoch_count, version)
    torch.save(net.state_dict(), save_path)

def validation():
    class_correct = list(0. for i in range(label_count))
    class_total = list(0. for i in range(label_count))
    total_acc = 0
    with torch.no_grad():
        for data in dataloaders[VAL]:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs, _ = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(label_count):
        print('Accuracy class_correctof %5s : %2d %%' % (
            image_datasets[VAL].classes[i], 100 * class_correct[i] / class_total[i]))

        total_acc =  total_acc + (100 * class_correct[i] / class_total[i])

    print('Accuracy total class_correct of  : %2d %%' % ( total_acc/label_count) )

def test():
    class_correct = list(0. for i in range(label_count))
    class_total = list(0. for i in range(label_count))
    total_acc = 0
    with torch.no_grad():
        for data in dataloaders[TEST]:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs, f = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(label_count):
        print('Accuracy of %5s : %2d %%' % (
            image_datasets[TEST].classes[i], 100 * class_correct[i] / class_total[i]))

        total_acc = total_acc + (100 * class_correct[i] / class_total[i])


    print('Accuracy total class : %2d %%' % (total_acc / label_count))

def model_loader():
    model_folder = '/home/mll/v_mll3/OCR_data/VGG_character/model'
    model_list = os.listdir(model_folder)
    print("---------------Select model ------------")
    for i in range(0 , len(model_list)):
        print("{} : {}".format(i,  model_list[i]))
    num = input()
    model_dir = model_folder+ '/'+ model_list[int(num)]
    #print(model_dir)

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

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        transform=data_transforms[x]
    )
    for x in [TRAIN, VAL, TEST]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=8,        #origin ver batch= 4 | ver1 batch=2 | ver2  batch=8  | ver batch=4
        shuffle=True, num_workers=4
    )
    for x in [TRAIN, VAL, TEST]
}







dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}

for x in [TRAIN, VAL, TEST]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))

print("Classes: ")
class_names = image_datasets[TRAIN].classes
label_count = len(image_datasets[TRAIN].classes)
print(image_datasets[TRAIN].classes)

s = 't'
while(s != "1" or s !="2"):
    print("Please command \n [ ( 1 ) train the model |  ( 2 ) load pre-trained model | (3) test_sample_case ] ")
    s = input()
    if s== "1":

        # need to set epoch, version info.

        train()
        print("-----------------------")
        validation()
        print("-----------------------")
        test()
        break


    elif s=="2":
        model_loader()
        net = Net()
        net.load_state_dict(torch.load(save_path))
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


    elif s=="3":
        model_loader()
        net = Net()
        net.load_state_dict(torch.load(save_path))
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




#해결필요함.
print("Test in real case")
sample_case_path = '/home/mll/v_mll3/OCR_data/VGG_character/Skeletonize_DNN/sample_test'




def image_loader(loader, image_name):

    image = Image.open(image_name)
    image = loader(image)
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

data_transforms2 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
print(net(image_loader(data_transforms2, sample_case_path+"/Test/1.jpg")))
'''
with torch.no_grad():

    images = images.cuda()
    labels = labels.cuda()

    outputs, f = net(images)
    _, predicted = torch.max(outputs, 1)
    c = (predicted == labels).squeeze()
    for i in range(len(labels)):
        label = labels[i]

    #print(images)
    print("target :  ",labels)
    #print("recognition :  ", _)
    #print(c)
    print("Recognition  : " , class_names[predicted])


'''


# functions to show an image

print("Done!")