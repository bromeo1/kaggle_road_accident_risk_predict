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
import streamlit as st
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

ave=[ 0.4887, 46.1126,  1.1880,  0.3336,  0.3310,  0.3550,  0.3011,  0.3505,
         0.3032]
std=[ 0.2726, 15.7885,  0.8960,  0.4715,  0.4706,  0.4785,  0.4588,  0.4771,
         0.4596]
ave=torch.tensor(ave,dtype=torch.float32)
std=torch.tensor(std,dtype=torch.float32)
st.write("please select the variables to predict the accident risk")
#select speed_limit
speed_limit=st.slider('select a speed limit',
                      25.0,70.0)
#select curvature
curvature=st.slider('select a curvature ',0.0,1.0)
#select number of reported accidents
num_reported_accidents=st.selectbox('select a the number of reported accidents on the road',
                                    (1,2,3,4,5,6,7,8,9,10))
#select road_type
road_type=st.selectbox('select a road type',
                       ('highway','urban','rural'))
#select lighting condition
lighting=st.selectbox('select a lighting condition',('dim','night','daylight'))
# select weather condition
weather=st.selectbox('select a weather condition',('rain','foggy','clear'))
#predict
if st.button('Predict accident risk'):
    X=torch.zeros(9,dtype=torch.float32)
    X[0]=curvature
    X[1]=speed_limit
    X[2]=num_reported_accidents
    if(road_type=='rural'):
        X[3]=1
    elif(road_type=='urban'):
        X[4]=1
    if(lighting=='dim'):
        X[5]=1
    elif(lighting=='night'):
        X[6]=1
    if(weather=='foggy'):
        X[7]=1
    elif(weather=='rain'):
        X[8]=1
    st.write(X)
    X = (X - ave) / std
    X=X.view(-1,9)
    model=torch.load('model1.pth')
    model.eval()
    with torch.no_grad():
        risk=model(X).item()
    st.success(f"Predicted accident risk: {risk:.2f}")
