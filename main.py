
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os, sys
import shutil
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from time import sleep
from tqdm import tqdm
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self, file):
        self.features=pd.read_csv(file)
        self.features = self.features.drop(self.features.columns[3], axis=1)
        self.features = self.features.drop(self.features.columns[7], axis=1)
        self.features["Embarked"] = self.features["Embarked"].replace({"Q":0, "S":1, "C":2})
        self.features["Sex"] = self.features["Sex"].replace({"male":1, "female":0})

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        passenger = self.features.iloc[index]
        return passenger[1],passenger[2:]

training_data = MyDataSet( file = "titanic/train.csv")
training_data = DataLoader(training_data, batch_size=16, shuffle=True)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(8, 100), nn.ReLU(),
            nn.Linear(100, 1), nn.Sigmoid()
        )

    def forward(self,x):
        x = self.model(x)
        return x
    
model = MyModel()

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(1,5):
    training_loss = 0

    model.train()
    
    for batch in :
        optimizer.zero_grad()
        
