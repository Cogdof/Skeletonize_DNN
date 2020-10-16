# -*- coding: utf-8 -*-
"""
This Script contains the default and Spinal VGG code for EMNIST(Letters).
This code trains both NNs as two different models.
This code randomly changes the learning rate to get a good result.
@author: Dipu
"""

import torch
import torchvision
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import os

'''
------------------------------
[Lastest update]
2020.10.16 fri 

[version]
ver 5.0 Data : Balanced(47 classes) |  Acc  VGG-5 :90.8 , spiral : 90.7  
ver 5.1 Data : byclasee(62 classes)


Dataset - 1 http://www.robots.ox.ac.uk/~vgg/data/text/#sec-chars
@InProceedings{Jaderberg14,
  author       = "Max Jaderberg and Andrea Vedaldi and Andrew Zisserman",
  title        = "Deep Features for Text Spotting",
  booktitle    = "European Conference on Computer Vision",
  year         = "2014",
}
---------------
'''


num_epochs = 200
batch_size_train = 100
batch_size_test = 1000
learning_rate = 0.005
momentum = 0.5
log_interval = 500

model_ver = 5.1

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.EMNIST('/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/dataset/files/', split='byclass', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.RandomPerspective(),
                                    torchvision.transforms.RandomRotation(10, fill=(0,)),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.EMNIST('/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/dataset/files/', split='byclass', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_test, shuffle=True)


realcase_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(root='/home/mll/v_mll3/OCR_data/dataset/MNIST_dataset/EMNIST_byclass/Validation',
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.Grayscale(1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])), batch_size=10, shuffle=True

)

custom_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
custom_dataset = torchvision.datasets.ImageFolder('/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/dataset/data/Validation',
                                                  transform= custom_transform)
custom_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=10, shuffle = True)



examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)

print("----------------")
examples2 = enumerate(realcase_loader)
batch_idx2, (example_data2, example_targets2) = next(examples2)
print(example_data2.shape)

print("label : ", realcase_loader.dataset.classes)

import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
fig


class VGG(nn.Module):
    """
    Based on - https://github.com/kkweon/mnist-competition
    from: https://github.com/ranihorev/Kuzushiji_MNIST/blob/master/KujuMNIST.ipynb
    """

    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s

    def three_conv_pool(self, in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s

    def __init__(self, num_classes=62):
        super(VGG, self).__init__()
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


Half_width = 128
layer_width = 128


class SpinalVGG(nn.Module):
    """
    Based on - https://github.com/kkweon/mnist-competition
    from: https://github.com/ranihorev/Kuzushiji_MNIST/blob/master/KujuMNIST.ipynb
    """

    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s

    def three_conv_pool(self, in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s

    def __init__(self, num_classes=62):
        super(SpinalVGG, self).__init__()
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(Half_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True), )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True), )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True), )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True), )
        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(layer_width * 4, num_classes), )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)

        x1 = self.fc_spinal_layer1(x[:, 0:Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([x[:, Half_width:2 * Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([x[:, 0:Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([x[:, Half_width:2 * Half_width], x3], dim=1))

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        x = self.fc_out(x)

        return F.log_softmax(x, dim=1)

#-------------------------------------------------------------------------
device = 'cuda'
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

#-------------------------------------------------------------

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Train the model
total_step = len(train_loader)
curr_lr1 = learning_rate

curr_lr2 = learning_rate



def train():
    model1 = VGG().to(device)
    model2 = SpinalVGG().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)

    best_accuracy1 = 0
    best_accuracy2 = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model1(images)
            loss1 = criterion(outputs, labels)

            # Backward and optimize
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            outputs = model2(images)
            loss2 = criterion(outputs, labels)

            # Backward and optimize
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            if i == 499:
                print("Ordinary Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss1.item()))
                print("Spinal Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss2.item()))

        # Test the model
        model1.eval()
        model2.eval()
        with torch.no_grad():
            correct1 = 0
            total1 = 0
            correct2 = 0
            total2 = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model1(images)
                _, predicted = torch.max(outputs.data, 1)
                total1 += labels.size(0)
                correct1 += (predicted == labels).sum().item()

                outputs = model2(images)
                _, predicted = torch.max(outputs.data, 1)
                total2 += labels.size(0)
                correct2 += (predicted == labels).sum().item()

            if best_accuracy1 >= correct1 / total1:
                curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
                update_lr(optimizer1, curr_lr1)
                print('Test Accuracy of NN: {} % Best: {} %'.format(100 * correct1 / total1, 100 * best_accuracy1))
            else:
                best_accuracy1 = correct1 / total1
                net_opt1 = model1
                print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct1 / total1))

            if best_accuracy2 >= correct2 / total2:
                curr_lr2 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
                update_lr(optimizer2, curr_lr2)
                print(
                    'Test Accuracy of SpinalNet: {} % Best: {} %'.format(100 * correct2 / total2, 100 * best_accuracy2))
            else:
                best_accuracy2 = correct2 / total2
                net_opt2 = model2
                print('Test Accuracy of SpinalNet: {} % (improvement)'.format(100 * correct2 / total2))

            model1.train()
            model2.train()

    save_path = "/home/mll/v_mll3/OCR_data/VGG_character/model/vgg_spinal/vgg5_{}_.pth".format(model_ver)
    torch.save(model1.state_dict(), save_path)

    save_path = "/home/mll/v_mll3/OCR_data/VGG_character/model/vgg_spinal/spinal_{}_.pth".format(model_ver)
    torch.save(model2.state_dict(), save_path)

def eval():
    model1 = VGG()
    model2 = SpinalVGG()

    model_list = os.listdir("/home/mll/v_mll3/OCR_data/VGG_character/model/vgg_spinal/")

    for save_model in model_list:
        print(save_model)

    load_model = input()

    save_path1 = "/home/mll/v_mll3/OCR_data/VGG_character/model/vgg_spinal/vgg5_{}_.pth".format(load_model)
    save_path2 = "/home/mll/v_mll3/OCR_data/VGG_character/model/vgg_spinal/spinal_{}_.pth".format(load_model)

    model1.load_state_dict(torch.load(save_path1))
    model2.load_state_dict(torch.load(save_path2))

    model1.to(device)
    model2.to(device)

    print(device)

    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)

    # Train the model

    best_accuracy1 = 0
    best_accuracy2 = 0
    # Test the model
    model1.eval()
    model2.eval()
    with torch.no_grad():
        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model1(images)
            _, predicted = torch.max(outputs.data, 1)
            total1 += labels.size(0)
            correct1 += (predicted == labels).sum().item()

            outputs = model2(images)
            _, predicted = torch.max(outputs.data, 1)
            total2 += labels.size(0)
            correct2 += (predicted == labels).sum().item()

        if best_accuracy1 >= correct1 / total1:
            curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
            update_lr(optimizer1, curr_lr1)
            print('Test Accuracy of NN: {} % Best: {} %'.format(100 * correct1 / total1, 100 * best_accuracy1))
        else:
            best_accuracy1 = correct1 / total1
            net_opt1 = model1
            print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct1 / total1))

        if best_accuracy2 >= correct2 / total2:
            curr_lr2 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
            update_lr(optimizer2, curr_lr2)
            print(
                'Test Accuracy of SpinalNet: {} % Best: {} %'.format(100 * correct2 / total2, 100 * best_accuracy2))
        else:
            best_accuracy2 = correct2 / total2
            net_opt2 = model2
            print('Test Accuracy of SpinalNet: {} % (improvement)'.format(100 * correct2 / total2))

def eval_outputcase(output_dataset):
    print("--- eval_realcase ---")
    model1 = VGG()
    model2 = SpinalVGG()

    model_list = os.listdir("/home/mll/v_mll3/OCR_data/VGG_character/model/vgg_spinal/")

    for save_model in model_list:
        print(save_model)

    load_model = input()

    save_path1 = "/home/mll/v_mll3/OCR_data/VGG_character/model/vgg_spinal/vgg5_{}_.pth".format(load_model)
    save_path2 = "/home/mll/v_mll3/OCR_data/VGG_character/model/vgg_spinal/spinal_{}_.pth".format(load_model)

    model1.load_state_dict(torch.load(save_path1))
    model2.load_state_dict(torch.load(save_path2))

    model1.to(device)
    model2.to(device)

    print(device)


    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)

    # Train the model

    best_accuracy1 = 0
    best_accuracy2 = 0
    # Test the model
    model1.eval()
    model2.eval()
    with torch.no_grad():
        print("-")
        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        for images, labels in output_dataset:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model1(images)
            _, predicted = torch.max(outputs.data, 1)
            total1 += labels.size(0)
            correct1 += (predicted == labels).sum().item()

            outputs = model2(images)
            _, predicted = torch.max(outputs.data, 1)
            total2 += labels.size(0)
            correct2 += (predicted == labels).sum().item()

        if best_accuracy1 >= correct1 / total1:
            curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
            update_lr(optimizer1, curr_lr1)
            print('Test Accuracy of NN: {} % Best: {} %'.format(100 * correct1 / total1, 100 * best_accuracy1))
        else:
            best_accuracy1 = correct1 / total1
            net_opt1 = model1
            print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct1 / total1))

        if best_accuracy2 >= correct2 / total2:
            curr_lr2 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
            update_lr(optimizer2, curr_lr2)
            print(
                'Test Accuracy of SpinalNet: {} % Best: {} %'.format(100 * correct2 / total2, 100 * best_accuracy2))
        else:
            best_accuracy2 = correct2 / total2
            net_opt2 = model2
            print('Test Accuracy of SpinalNet: {} % (improvement)'.format(100 * correct2 / total2))



print("1: train |  2: load model  |  3: realcase eval")
model_choose = input()

if model_choose=="1":
    train()

elif model_choose=="2":
    eval()

elif model_choose=="3":
    eval_outputcase(realcase_loader)

else:
    print("wrong input")

print("Done!")
