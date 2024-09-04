
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

from sklearn.ensemble import RandomForestClassifier
os.chdir('C:/Users/iacop/Documents/GitHub/Kaggle1/Titanic_Kaggle')
train_data =pd.read_csv("titanic/train.csv")
test_data =pd.read_csv("titanic/test.csv")

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
print(X.head())
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")