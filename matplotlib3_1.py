import numpy as np
import pandas as pd

#Задание1

from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()
data = boston["data"]
data.shape
feature_names = boston["feature_names"]
feature_names
target = boston["target"]
target[:10]
X = pd.DataFrame(data, columns=feature_names)
X.head()
X.info()
y = pd.DataFrame(target, columns=["price"])
y.info()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred.shape
check_test = pd.DataFrame({
 "y_test": y_test["price"],
 "y_pred": y_pred.flatten(),
})
check_test.head(10)
check_test["error"] = check_test["y_pred"] - check_test["y_test"]
check_test.head()
from sklearn.metrics import r2_score
r2_score_1=r2_score(check_test["y_pred"], check_test["y_test"])
r2_score_1

#Задание2
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)
model.fit(X_train, y_train.values[:, 0])
y_pred = model.predict(X_test)
y_pred.shape
check_test = pd.DataFrame({
 "y_test": y_test["price"],
 "y_pred": y_pred.flatten(),
})
check_test.head(10)
r2_score_2=r2_score(check_test["y_pred"], check_test["y_test"])
r2_score_2
r2_score_1<r2_score_2
#Дерево решений работает лучше, чем линейная регрессия, т.к значение ближе к единце

?RandomForestRegressor
print(model.feature_importances_)
model.feature_importances_.sum()
max_value_idx1=model.feature_importances_.argmax()
max_value_idx1

max_value_idx2=0
max_value=model.feature_importances_[max_value_idx2]
for i in range(model.n_features_):
    if max_value < model.feature_importances_[i] and i!=max_value_idx1:
        max_value=model.feature_importances_[i]
        max_value_idx2=i
print(max_value_idx2)
