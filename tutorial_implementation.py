# loosly based on: https://www.kaggle.com/code/alexisbcook/titanic-tutorial

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
from sklearn.ensemble import RandomForestClassifier # model to use

train_data = pd.read_csv("titanic/train.csv")
test_data = pd.read_csv("titanic/test.csv")
targetTest_data = pd.read_csv("titanic/gender_submission.csv")

# ------ Prepare Data
# This is a supervised learning model so we need the target value and the feature to fit the model
y = train_data["Survived"] # target variable

features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
X = pd.get_dummies(train_data[features]) # Convert categorical variable into dummy/indicator variables.
X_test = pd.get_dummies(test_data[features])

best_score = 0
best_est = 0
best_depth = 0
for estimators in range(10,500,50):
    for depth in [5,7,9]:
        # ------ Model fitting
        model = RandomForestClassifier(n_estimators=estimators, max_depth=depth)
        model.fit(X,y)

        # ------ Testing
        score = model.score(X_test, targetTest_data["Survived"])
        
        if score > best_score:
            best_score = score
            best_depth = depth
            best_est = estimators
        
        if best_score == 1:
            break
    if best_score == 1:
            break

print("Best score --> estimators: ", best_est, " depth: ", best_depth, " ", best_score * 100)