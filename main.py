
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os, sys
import shutil
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from time import sleep
from tqdm import tqdm
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, file):
        self.features=pd.read_csv(file)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features.iloc[index,1],self.features[index,2:]


train_data=pd.read_csv("titanic/train.csv")
train_data.head()

