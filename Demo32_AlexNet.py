#!/usr/bin/python
#encoding:utf-8
"""
@author:cheney
@file:Demo32_AlexNet.py
@time:2019/9/20 19:53
"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.optim as optim
import time

#定义一个数据预处理的函数
def data_tf(x):
    x=np.array(x,dtype=np.float32)/255
    x=(x-0.5)/0.5
    x=x.transpose((2,0,1))#原先是[32,32,3],将channel放在第一维，pytorch要求这样做
    x=torch.from_numpy(x)
    return x


#获得数据集
train_set=CIFAR10('./data',train=True,transform=data_tf,download=False)
test_set=CIFAR10('./data',train=False,transform=data_tf,download=False)
train_data=DataLoader(train_set,batch_size=64,shuffle=True)
test_data=DataLoader(test_set,batch_size=128,shuffle=False)


#数据集大小
train_size=len(train_set)
test_size=len(test_set)

#定义device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#定义论文中的AlexNet模型
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        #第一层是5x5的卷积核，input_channels=3,output_channels=64,strride=1，padding=0
        self.conv1=nn.Sequential(
            nn.Conv2d(3,64,5),
            nn.ReLU(inplace=True)
        )
        #第二层是3 x 3的池化，strride=2,padding=0,
        #注意池化层的size计算 也是（w-k+2p）/s+1
        self.max_pool1=nn.MaxPool2d(3,2)
        #第三层是5 x5卷积，input=64,output=64,strride=1
        self.conv2=nn.Sequential(
            nn.Conv2d(64,64,5,),
            nn.ReLU(inplace=True)
        )
        #第四层是3 x3的池化，strride=2
        self.max_pool2=nn.MaxPool2d(3,2)
        #第五层是全连接层，input=1024,output=384
        self.fc1=nn.Sequential(
            nn.Linear(1024,384),
            nn.ReLU(inplace=True)
        )
        #第六层全连接层，input=384,output=192
        self.fc2=nn.Sequential(
            nn.Linear(384,192),
            nn.ReLU(inplace=True)
        )
        #第七层是全连接层，input=192,output=10
        self.fc3=nn.Linear(192,10)
        self.dropout=nn.Dropout(0.4)

    def forward(self, x):
        x=self.conv1(x)
        x=self.max_pool1(x)
        x=self.conv2(x)
        x=self.max_pool2(x)
        x = self.dropout(x)

        #将矩阵拉平
        x=x.view(x.shape[0],-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x


#生成模型
alexnet=AlexNet().to(device)

#验证一下网络的结构是否正确
# input_demo=Variable(torch.zeros(1,3,32,32))
# output_demo=alexnet(input_demo)


#定义超参
learning_rate=1e-3
epochs=20


#定义优化器
optimizer=optim.Adam(alexnet.parameters(),lr=learning_rate)
#分类问题，使用交叉熵损失函数
criterion=nn.CrossEntropyLoss().to(device)

#开始训练
for epoch in range(epochs):
    train_loss=0
    train_acc=0
    alexnet.train()
    start=time.time()
    for img,label in train_data:
        img=Variable(img).to(device)
        label=Variable(label).to(device)

        #训练
        out=alexnet(img)
        optimizer.zero_grad()
        #交叉熵损失函数中已经考虑了多分类问题
        loss=criterion(out,label)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*label.shape[0]/train_size

        #计算准确度
        pred=torch.argmax(out,dim=1)
        train_acc+=(pred==label).sum().float()/train_size

    alexnet.eval()
    test_loss=0
    test_acc=0
    for img,label in test_data:
        img=Variable(img).to(device)
        label=Variable(label).to(device)

        #评估
        out=alexnet(img)
        loss=criterion(out,label)
        test_loss+=loss.item()*label.shape[0]/test_size

        #计算准确度
        pred=torch.argmax(out,dim=1)
        test_acc+=(pred==label).sum().float()/test_size
    end = time.time()
    print('Epoch:{},Train Loss:{:.6f},Train Acc:{:.6f},Valid Loss:{:.6f},Valid Acc:{:.6f},Time:{:.3f}s'.format(
        epoch,train_loss,train_acc,test_loss,test_acc,(end-start)
    ))