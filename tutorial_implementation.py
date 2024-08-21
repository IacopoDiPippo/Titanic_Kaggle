# Based on: https://www.kaggle.com/code/alexisbcook/titanic-tutorial

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
from sklearn.ensemble import RandomForestClassifier # model to use

train_data = pd.read_csv("titanic/train.csv")
test_data = pd.read_csv("titanic/test.csv")
targetTest_data = pd.read_csv("titanic/gender_submission.csv")

# ------ Prepare Data
# This is a supervised learning model so we need the target value and the feature to fit the model
y = train_data["Survived"] # target variable

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features]) # Convert categorical variable into dummy/indicator variables.
X_test = pd.get_dummies(test_data[features])

# ------ Model fitting
model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X,y)

# ------ Testing
print(model.score(X_test, targetTest_data["Survived"]))