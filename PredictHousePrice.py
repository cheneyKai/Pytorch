#!/usr/bin/python
#encoding:utf-8
"""
@author:cheney
@file:Demo30_PredictHousePrice.py
@time:2019/9/19 22:13
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader

np.set_printoptions(suppress=True)

#设置设备
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#读入数据
housing_data=[]
with open('data/housing.txt','r') as fin:
    for line in fin.readlines():
        housing_data.append(line.strip().split())

#转化为numpy类型的数组
housing_data_np=np.array(housing_data,dtype=np.float32)
#!/usr/bin/python
#encoding:utf-8
"""
@author:cheney
@file:Demo30_PredictHousePrice.py
@time:2019/9/19 22:13
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

#设置设备
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#读入数据
housing_data=[]
with open('data/housing.txt','r') as fin:
    for line in fin.readlines():
        housing_data.append(line.strip().split())

#转化为numpy类型的数组
housing_data_np=np.array(housing_data,dtype=np.float32)

#转化成toser
dataset=torch.from_numpy(housing_data_np).float()

#随机划分训练集和测试集
train_size=int(dataset.shape[0]*0.8)
test_size=dataset.shape[0]-train_size
train_set,test_set=data.random_split(dataset=dataset,lengths=[train_size,test_size])
train_data=DataLoader(train_set,batch_size=64,shuffle=True)
test_data=DataLoader(test_set,batch_size=128,shuffle=True)
#定义模型
class DNN(nn.Module):
    def __init__(self):
        super(DNN,self).__init__()
        self.layer1=nn.Linear(13,128)
        self.layer2=nn.Linear(128,256)
        self.layer3=nn.Linear(256,64)
        self.layer4=nn.Linear(64,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x=self.layer1(x)
        x=self.relu(x)
        x=self.layer2(x)
        x=self.relu(x)
        x=self.layer3(x)
        x=self.relu(x)
        x=self.layer4(x)
        

        return x

net=DNN().to(device)
lr=1e-3
epochs=101
#定义优化器，使用MSE
criterion=nn.MSELoss().to(device)
optimizer=torch.optim.Adam(net.parameters(),lr=lr)

for epoch in range(epochs):
    train_loss=0
    for min_data in train_data:
        x=min_data[:,:-1]
        y=min_data[:,-1]
        x=Variable(x).to(device)
        y=Variable(y).to(device)
        out=net(x)
        out=out.squeeze(1)
        optimizer.zero_grad()
        loss=criterion(out,y)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    if epoch%50==0:
        test_loss=0

        for min_data in test_data:
            x_t = min_data[:, :-1]
            y_t = min_data[:, -1]
            x_t= Variable(x_t).to(device)
            y_t= Variable(y_t).to(device)
            out = net(x_t)
            out = out.squeeze(1)
            loss = criterion(out, y_t)
            test_loss += loss.item()

        print('epoch:{},Train Loss:{:.5f},Test Loss:{:.5f}'.format(epoch, train_loss / y.shape[0],test_loss/y_t.shape[0]))



#最后画出预测与实际对比图
real=[]
pred=[]
for min_data in test_data:
    x_t = min_data[:, :-1]
    y_t = min_data[:, -1]
    x_t= Variable(x_t).to(device)
    y_t= Variable(y_t).to(device)
    real.append(y_t.cpu().detach().numpy().tolist())
    out = net(x_t)
    out = out.squeeze(1)
    pred.append(out.cpu().detach().numpy().tolist())
    loss = criterion(out, y_t)
    test_loss += loss.item()

real=real[0]
pred=pred[0]
x=np.linspace(-10,10,102)

plt.plot(x,real,label='real price')
plt.plot(x,pred,label='pred price')
plt.legend(loc='best')
plt.show()
