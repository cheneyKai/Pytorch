#!/usr/bin/python
#encoding:utf-8
"""
@author:cheney
@file:Demo33_VGGNet.py
@time:2019/9/21 12:33
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision.datasets import CIFAR10
from  torch.utils.data import DataLoader
import time


#定义设备
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#数据标准化
def data_tf(x):
    x=np.array(x,dtype=np.float32)
    x=x/255.
    x=(x-0.5)/0.5
    x=x.transpose(2,0,1)#要把channel放在最前面
    x=torch.from_numpy(x)
    return x



#获取数据
train_set=CIFAR10('./data',train=True,transform=data_tf,download=False)
test_set=CIFAR10('./data',train=False,transform=data_tf,download=False)

#加载数据集
train_data=DataLoader(train_set,batch_size=64,shuffle=True)
test_data=DataLoader(test_set,batch_size=128,shuffle=False)

#数据集大小
train_size=len(train_set)
test_size=len(test_set)

#定义一个vgg_block，三个参数（模型层数，输入通道数，输出通道数）
def vgg_block(num_convs,in_channels,out_channels):
    #定义第一层
    net=[
        nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),#padding=1使得图片尺寸不变
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels,1e-3),
        nn.Dropout(0.5)
    ]
    # 定义后面的层
    for i in range(num_convs-1):
        net.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        net.append(nn.ReLU(inplace=True))

    #定义池化层
    net.append(nn.MaxPool2d(2,2))
    #list变量前面加*代表将list分解成独立的变量。
    return nn.Sequential(*net)

#打印出一个模型看看
# block_demo=vgg_block(3,64,128)
#定义一个输入[1,64,300,300]
# input=Variable(torch.zeros(1,64,300,300))
# out=block_demo(input)
# print(out.shape)

#定义一个函数对VGG堆叠
def vgg_stack(num_convs,channels):
    net=[]
    for n,c in zip(num_convs,channels):
        in_c=c[0]
        out_c=c[1]
        net.append(vgg_block(n,in_c,out_c))
    return nn.Sequential(*net)


#创建一个具有8个卷积层的简单VGG
vgg_net=vgg_stack((1,1,2,2,2),channels=((3,64),(64,128),(128,256),(256,512),(512,512)))

#测试一下
# input_demo=Variable(torch.zeros(1,3,32,32))
# out=vgg_net(input_demo)

#最后需要加上几层全连接层得到想要的分类

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.feature=vgg_net
        self.fc=nn.Sequential(
            nn.Linear(512,100),
            nn.ReLU(inplace=True),
            nn.Linear(100,10)
        )

    def forward(self, x):
        x=self.feature(x)
        x=x.view(x.shape[0],-1)
        x=self.fc(x)
        return x

#学习率
learning_rate=1e-3
epochs=20

#生成模型
net=VGG().to(device)

#优化器
optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)

#损失函数，分类-》交叉熵
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



