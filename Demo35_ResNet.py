#!/usr/bin/python
#encoding:utf-8
"""
@author:cheney
@file:Demo35_ResNet.py
@time:2019/9/21 20:24
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transform
import time

#数据格式转换
def data_tf(x):
    x=x.resize((96,96),2)
    x=np.array(x,dtype=np.float32)/255.
    x=(x-0.5)/0.5
    x=x.transpose((2,0,1))
    x=torch.from_numpy(x)
    return x

#数据增强,除了batch的维度，其他三个维度
def train_tf(x):
    img_aug=transform.Compose([
        transform.Resize(120),
        transform.RandomHorizontalFlip(),
        transform.RandomCrop(96),
        transform.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5),
        transform.ToTensor(),
        transform.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

    ])
    x=img_aug(x)
    return x

def test_tf(x):
    img_aug=transform.Compose([
        transform.Resize(96),
        transform.ToTensor(),
        transform.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    x=img_aug(x)
    return x


#设备
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#数据
train_set=CIFAR10('./data',train=True,transform=train_tf,download=False)
test_set=CIFAR10('./data',train=False,transform=test_tf,download=False)

train_data=DataLoader(train_set,batch_size=64,shuffle=True)
test_data=DataLoader(test_set,batch_size=128,shuffle=True)

#长度
train_size=len(train_set)
test_size=len(test_set)



#定义一个3x3的卷积，作为基本结构
def conv3x3(in_channel,out_channel,stride=1):
    return nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)


#定义残差结构
class ResidualBlock(nn.Module):
    def __init__(self,in_channel,out_channel,same_shape=True):
        super(ResidualBlock, self).__init__()
        self.same_shape=same_shape
        stride=1 if self.same_shape else 2

        #第一个卷积层
        self.conv1=conv3x3(in_channel,out_channel,stride=stride)
        self.bn1=nn.BatchNorm2d(out_channel)

        self.conv2=conv3x3(out_channel,out_channel)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU(inplace=True)
        if not self.same_shape:
            self.conv3=nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=stride)


    def forward(self, x):
        out=self.conv1(x)
        out=self.relu(self.bn1(out))

        out=self.conv2(out)
        out=self.relu(self.bn2(out))

        if not self.same_shape:
            x=self.conv3(x)

        return self.relu(x+out)


#测试一下残差网络
# test_net=ResidualBlock(32,32)
# test_x=Variable(torch.zeros(1,32,96,96))
# out=test_net(test_x)
# print(out.shape)

#输入不同的形状
# test_net=ResidualBlock(3,32,False)
# test_x=Variable(torch.zeros(1,3,96,96))
# out=test_net(test_x)
# print(out.shape)

#实现ResNet，他是ResidualBlock的堆叠
class ResNet(nn.Module):
    def __init__(self,in_channel,num_classes,verbose=False):
        super(ResNet,self).__init__()
        self.verbose=verbose

        self.block1=nn.Conv2d(in_channel,out_channels=64,kernel_size=7,stride=2)

        self.block2=nn.Sequential(
            nn.MaxPool2d(3,2),
            ResidualBlock(64,64),
            ResidualBlock(64,64)
        )

        self.block3=nn.Sequential(
            ResidualBlock(64,128,False),
            ResidualBlock(128,128)

        )
        self.block4=nn.Sequential(
            ResidualBlock(128,256,False),
            ResidualBlock(256,256)
        )
        self.block5=nn.Sequential(
            ResidualBlock(256, 512, False),
            ResidualBlock(512, 512),
            nn.AvgPool2d(3)

        )

        self.classifier=nn.Linear(512,num_classes)


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


#输出每个block之后的大小

# test_net=ResNet(3,10,verbose=True)
# test_x=Variable(torch.zeros(1,3,96,96))
# out=test_net(test_x)
# print('output:{}'.format(out.shape))

#学习率，
learning_rate=1e-3
epochs=20


#定义网络
net=ResNet(3,10).to(device)

#优化器
optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)

criterion=nn.CrossEntropyLoss().to(device)

#开始训练
for epoch in range(epochs):
    train_loss=0
    train_acc=0
    net=net.train()
    start=time.time()
    for img,label in train_data:
        img=Variable(img).to(device)
        label=Variable(label).to(device)
        #清空梯度
        optimizer.zero_grad()
        out=net(img)
        #计算loss
        loss=criterion(out,label)
        train_loss+=loss.item()*label.shape[0]/train_size
        pred=torch.argmax(out,dim=1)
        train_acc+=(pred==label).sum().float()/train_size
        #反向传播
        loss.backward()
        optimizer.step()

    test_loss=0
    test_acc=0
    net=net.eval()
    for img,label in test_data:
        img=Variable(img).to(device)
        label=Variable(label).to(device)
        out=net(img)
        #计算loss
        loss=criterion(out,label)
        test_loss+=loss.item()*label.shape[0]/test_size
        pred=torch.argmax(out,dim=1)
        test_acc+=(pred==label).sum().float()/test_size
    end=time.time()
    print('Epoch:{},Train Loss:{:.6f},Train Acc:{:.6f},Valid Loss:{:.6f},Valid Acc:{:.6f},Time:{:.3f}s'.format(
        epoch,train_loss,train_acc,test_loss,test_acc,(end-start)
    ))
