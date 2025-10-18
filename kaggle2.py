#%%
import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import torch.optim as optim
from pyexpat import model
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#%%
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
print(train_data.head())
# print(test_data.head())

#%%
#检查缺失值
print(type(test_data))
print(train_data.isnull().sum())
print(test_data.isnull().sum())
#数据集很好
#%%
#统计数值类型
round(train_data.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]),4)
#describe默认情况下仅针对数值类型计算
#%%
#统计对象类型
train_data.describe(include=['object'])
#%%
#统计bool类型
train_data.describe(include=['bool'])
#%%
#分析curvature属性
#用直方图统计
#num_lanes发现直接关系不大,基本无关，无论在低事故率还是高事故率下均匀分布在四种取值下,在极端情况下有一定关系  ###可以不用
#curvature属性可以发现在事故率较低的数据中，他们大多数curvature值都很低，而高事故率的数据curvature很高，可以发现一个很强正相关的关系，强相关
#speed_limit,也是正相关，强相关
#num_reported_accidents 在低事故率下大多数曾被报道事故数都很少 ，有相关关系
# road_type 关系较弱，仅在高事故率下数据的road_type多为rural,关系稍弱，还是可以用
#lighting 高事故率条件下，数据为night的概率很高可以认为两者有关系
#weather可选，有一定相关性
#time_of_day 有比较弱的相关性  ###可以不用
#road_signs_present      ###可以不用
#public_road             ###关系也较弱二等弱
#holiday 相关性还行
#数值列筛掉 id, num_lanes,
#bool列筛掉 rsp,pr, hd,sc
#
x=train_data.loc[train_data['accident_risk']>0.8,'holiday']
for name in train_data.select_dtypes(include=['bool']).columns.tolist():
    print(train_data[['accident_risk', name]].corr(method='pearson'))
    print('\n')
x.value_counts().sort_index().plot(kind='bar')

plt.show()

#%%
#筛掉无关特征
useless_feature=['id','num_lanes','accident_risk','road_signs_present','public_road','holiday','school_season','time_of_day']
useless_feature2=['id','num_lanes','road_signs_present','public_road','holiday','school_season','time_of_day']
X_train=train_data.drop(useless_feature,axis=1)
X_test=test_data.drop(useless_feature2,axis=1)
y_train=train_data['accident_risk']
print(X_train.head())
#%%
#转换非数值类型为独热编码  新版本变为bool变量
X_train_encoded=pd.get_dummies(X_train,drop_first=True,dtype=np.float32)
X_test_encoded=pd.get_dummies(X_test,drop_first=True,dtype=np.float32)
print(X_train_encoded.head())
#%%
#bool删除
bool_cols=X_train_encoded.select_dtypes(include=['bool']).columns.tolist()
print(bool_cols)
X_train_encoded_boolto=X_train_encoded.drop(bool_cols,axis=1)
X_test_encoded_boolto=X_test_encoded.drop(bool_cols,axis=1)
print(X_train_encoded_boolto.shape)
#%%
X_train=X_train_encoded_boolto
X_test=X_test_encoded_boolto
X_train=torch.tensor(X_train.values,dtype=torch.float32)
X_test=torch.tensor(X_test.values,dtype=torch.float32)
y_train=torch.tensor(y_train.values,dtype=torch.float32)
print(X_train.shape)
#%%
print(X_test[0])
ave=torch.mean(X_train,dim=0)
std=torch.std(X_train,dim=0)
X_train=(X_train-ave)/std
X_test=(X_test-ave)/std
print(X_train[2])
#%%
torch.manual_seed(42)
#%%
#换划分训练集和测试集
indices=torch.randperm(X_train.shape[0])  #随机打乱
split_idx=int(X_train.shape[0]*0.8)
train_indices=indices[:split_idx]
verify_indices=indices[split_idx:]
X_train_split=X_train[train_indices]
X_verify_split=X_train[verify_indices]
y_train_split=y_train[train_indices]
y_verify_split=y_train[verify_indices]
print("X_train shape:", X_train.shape)
#%%
trainset=TensorDataset(X_train_split,y_train_split)
testset=TensorDataset(X_verify_split,y_verify_split)

#%%
trainloader=DataLoader(trainset,batch_size=32,shuffle=True)
testloader=DataLoader(testset,batch_size=32,shuffle=False)
#%%
#构造全连接神经网络，最后输出用sigmoid转为概率
#网络层次 输入：9 hidden1:32 hidden2:16 output:1
class FCNN(nn.Module):
    def __init__(self,input_size=9,hidden1_size=32,hidden2_size=16,output_size=1):
        super(FCNN,self).__init__()
        self.fc1=nn.Linear(input_size,hidden1_size)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(hidden1_size,hidden2_size)
        self.fc3=nn.Linear(hidden2_size,output_size)
        self.sigmoid=nn.Sigmoid()
    def forward(self,X):
        out=self.fc1(X)
        out=self.relu(out)
        out=self.fc2(out)
        out=self.relu(out)
        out=self.fc3(out)
        out=self.sigmoid(out)
        return out
    def train_model(self,dataloader,optimizer,criterion):
        self.train()
        for idx,(data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output=self.forward(data)
            output=output.view(-1)
            loss=criterion(output,target)
            loss.backward()
            optimizer.step()
            # if idx%100==0:
            #     print('loss:',loss.item())
    def predict(self,X):
        self.eval()
        output=[]
        with torch.no_grad():
            for data in X:
                output.append(self.forward(data))
        return output
    def evaluate(self,dataloader,criterion):
        self.eval()
        losssum=0
        with torch.no_grad():
            for idx,(data, target) in enumerate(dataloader):
                output=self.forward(data)
                output=output.view(-1)
                loss=criterion(output,target)
                losssum=losssum+loss.item()
        print('loss:',losssum/len(dataloader))


#%%
model=FCNN()
#%%
optimizer=optim.SGD(model.parameters(),lr=0.01)
criterion=nn.L1Loss()
#%%
#0.444
for epoch in range(100):
    model.train_model(trainloader,optimizer,criterion)
    model.evaluate(testloader,criterion)
#%%
for i in range(10):
    model.train_model(trainloader,optimizer,criterion)
#%%
predict=model.predict(X_test)
#%%
num_p=[t.item() for t in predict]


#%%
ps=pd.Series(num_p)
#%%
print(ps)
#%%
df=pd.DataFrame({
    'id': test_data['id'],
    'accident_risk': ps
})
#%%
print(df.head())
#%%
df.to_csv('predict.csv',index=False)
#%%
#可以优化的点数值归一化，有一项还是太大了相对其他项，全部归一化还是单个归一化
#可以试试加上正则化，
#由调整损失函数为L1Loss后结果更好猜测可能陷入了局部最优，可以试试其它优化方法
