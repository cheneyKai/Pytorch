#!/usr/bin/python
#encoding:utf-8
"""
@author:cheney
@file:Demo34_GoogLeNet.py
@time:2019/9/21 15:13
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as  np
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import time

#定义设备
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#定义数据转化格式
def data_tf(x):
    x=x.resize((96,96),2)#图片放大到96x96
    x=np.array(x,dtype=np.float32)
    x=x/255.
    x=(x-0.5)/0.5
    x=x.transpose((2,0,1))#将channel放在最前面
    x=torch.from_numpy(x)
    return x

#获取数据
train_set=CIFAR10('./data',train=True,transform=data_tf,download=False)
test_set=CIFAR10('./data',train=False,transform=data_tf,download=False)

#加载数据
train_data=DataLoader(train_set,batch_size=50,shuffle=True)
test_data=DataLoader(test_set,batch_size=64,shuffle=False)

#数据集长度
train_size=len(train_set)
test_size=len(test_set)

#定义基本结构：一个卷积，一个Relu和一个batchnorm作为一个基本结构

def conv_relu(in_channels,out_channels,kernel,stride=1,padding=0):
    layer=nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=kernel,stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels,eps=1e-3),
        nn.ReLU(inplace=True)
    )
    return layer

#inception模块
class Inception(nn.Module):
    def __init__(self,in_channel,out1_1,out2_1,out2_3,out3_1,out3_5,out4_1):
        super(Inception, self).__init__()
        #第一条线路
        self.branch1x1=conv_relu(in_channel,out1_1,kernel=1)
        #第二条线路
        self.branch3x3=nn.Sequential(
            conv_relu(in_channel,out2_1,kernel=1),
            conv_relu(out2_1,out2_3,kernel=3,padding=1)
        )
        #第三条线路
        self.branch5x5=nn.Sequential(
            conv_relu(in_channel,out3_1,kernel=1),
            conv_relu(out3_1,out3_5,kernel=5,padding=2)
        )
        #第四条线路
        self.branch_pool=nn.Sequential(
            nn.MaxPool2d(3,1,padding=1),
            conv_relu(in_channel,out4_1,1)
        )

    def forward(self, x):
        fc1=self.branch1x1(x)
        fc2=self.branch3x3(x)
        fc3=self.branch5x5(x)
        fc4=self.branch_pool(x)

        #拼接操作，由于shape都是[batchsize,?,32,32]，因此在第一个维度拼接
        output=torch.cat((fc1,fc2,fc3,fc4),dim=1)
        return output


# test_net=Inception(3,64,48,64,64,96,32)
# test_x=Variable(torch.zeros(1,3,32,32))
# out=test_net(test_x)
# print(out.shape)
#定义GoogLeNet

class GoogLeNet(nn.Module):
    def __init__(self,in_channel,num_classes,verbose=False):
        super(GoogLeNet, self).__init__()
        self.verbose=verbose

        self.block1=nn.Sequential(
            conv_relu(in_channel,out_channels=64,kernel=7,stride=2,padding=3),
            nn.MaxPool2d(3,2)
        )
        self.block2=nn.Sequential(
            conv_relu(64,64,kernel=1),
            conv_relu(64,192,kernel=3,padding=1),
            nn.MaxPool2d(3,2)
        )

        self.block3=nn.Sequential(
            Inception(192,64,96,128,16,32,32),
            Inception(256,128,128,192,32,96,64),
            nn.MaxPool2d(3,2)
        )

        self.block4=nn.Sequential(
            Inception(480,192,96,208,16,48,64),
            Inception(512,160,112,224,24,64,64),
            Inception(512,128,128,256,24,64,64),
            Inception(512,112,144,288,32,64,64),
            Inception(528,256,160,320,32,128,128),
            nn.MaxPool2d(3,2)
        )
        self.block5=nn.Sequential(
            Inception(832,256,160,320,32,128,128),
            Inception(832,384,182,384,48,128,128),
            nn.AvgPool2d(2)
        )
        self.classifier=nn.Linear(1024,num_classes)


    def forward(self, x):
        x=self.block1(x)
        if self.verbose:
            print('block1 output:{}'.format(x.shape))

        x=self.block2(x)
        if self.verbose:
            print('block2 output:{}'.format(x.shape))
        x=self.block3(x)
        if self.verbose:
            print('block3 output:{}'.format(x.shape))
        x=self.block4(x)
        if self.verbose:
            print('block4 output:{}'.format(x.shape))

        x=self.block5(x)
        if self.verbose:
            print('block5 output:{}'.format(x.shape))

        x=x.view(x.shape[0],-1)
        x=self.classifier(x)
        return x


# test_net=GoogLeNet(3,10,verbose=True)
# test_x=Variable(torch.zeros(1,3,96,96))
# out=test_net(test_x)
# print('the final output:{}'.format(out.shape))
# output:
# block1 output:torch.Size([1, 64, 23, 23])
# block2 output:torch.Size([1, 192, 11, 11])
# block3 output:torch.Size([1, 480, 5, 5])
# block4 output:torch.Size([1, 832, 2, 2])
# block5 output:torch.Size([1, 1024, 1, 1])
# the final output:torch.Size([1, 10])
#可以看到一个好的网络时通道的维度不断增加，输入的尺寸不断减少。

#定义学习率
learning_rate=1e-3
epochs=20

#定义网络
net=GoogLeNet(3,10).to(device)

#定义优化器
optimizer=optim.Adam(net.parameters(),lr=learning_rate)
#损失函数，分类-》交叉熵
criterion=nn.CrossEntropyLoss()

#开始训练

for epoch in range(epochs):
    train_loss=0
    train_acc=0
    net=net.train()
    start=time.time()

    for img,label in train_data:
        img=Variable(img).to(device)
        label=Variable(label).to(device)
        optimizer.zero_grad()
        out=net(img)
        loss=criterion(out,label)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*label.shape[0]/train_size
        pred=torch.argmax(out,dim=1)
        train_acc+=(pred==label).sum().float()/train_size

    test_loss=0
    test_acc=0
    net=net.eval()
    for img,label in test_data:
        img=Variable(img).to(device)
        label=Variable(label).to(device)
        out=net(img)
        loss=criterion(out,label)
        test_loss+=loss.item()*label.shape[0]/test_size
        pred=torch.argmax(out,dim=1)
        test_acc+=(pred==label).sum().float()/test_size

    end=time.time()
    print('Epoch:{},Train Loss:{:.6f},Train Acc:{:.6f},Valid Loss:{:.6f},Valid Acc:{:.6f},Time:{:.3f}s'.format(
        epoch, train_loss, train_acc, test_loss, test_acc, (end - start)
    ))


