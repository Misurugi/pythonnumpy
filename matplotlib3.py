
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
from sklearn.model_selection import train_test_split

DATASET_PATH = './creditcard.csv'

df=pd.read_csv (DATASET_PATH, sep= ',')
df["Class"].value_counts(normalize=True)
df.info()
df.isnull().astype(int).sum().astype(int)
pd.options.display.max_columns = 100
print(df.head(10))
x = df.drop("Class", axis=1)
print(x.head())
y = df["Class"]
print(y.head())
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=100, stratify=y)
x_train.shape
x_test.shape
y_train.shape
y_test.shape

parameters = [{'n_estimators': [10, 15],
'max_features': np.arange(3, 5),
'max_depth': np.arange(4, 7)}]

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
clf = GridSearchCV(estimator=RandomForestClassifier(random_state=100),
                   param_grid=parameters,
                   scoring='roc_auc',
                   cv=3
                   )
clf.fit(x_train, y_train)
clf.best_params_
print(clf.best_params_)
y_pred_proba = clf.predict_proba(x_test)
print(y_pred_proba[:10])
y_pred_proba = y_pred_proba[:, 1]
print(y_pred_proba[:5])
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_proba)
print(roc_auc_score(y_test, y_pred_proba))
