import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv

DATASET_PATH = './creditcard.csv'

df=pd.read_csv (DATASET_PATH, sep= ',')
df.head()
df.describe()
t=df['Class'].value_counts()
print(t)
"""""
df_class_info = pd.Series(t)
df_class_info.plot.bar()
plt.show()
df_class_info.plot(kind='bar', logy=True)
plt.show()
"""""
v1_class1=df.set_index('Class')['V1'].filter(like='1', axis=0)
v1_class1=v1_class1.reset_index()
v1_class1=v1_class1.drop('Class', axis=1)
v1_class1.head(), v1_class1.count()

v1_class0=df.set_index('Class')['V1'].filter(like='0', axis=0)
v1_class0=v1_class0.reset_index()
v1_class0=v1_class0.drop('Class', axis=1)
print(v1_class0.head(10))
print(v1_class0.count())

plt.hist(v1_class0['V1'], bins=20, color='grey', edgecolor='black', density = True, orientation='horizontal')
plt.hist(v1_class1['V1'], bins=20, color='red', edgecolor='black', density = True, orientation='horizontal')
plt.plot()
plt.xlabel('Class')
plt.legend(labels=['Class 0', 'Class 1'])
plt.show()