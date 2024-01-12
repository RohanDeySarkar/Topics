from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import accuracy_score
import numpy as np

X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])

y = [0,1,0,1,0,1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.1, random_state=3) #min_samples_leaf -> leaf must contain atleast 10% of the data

dt.fit(X_test, y_test)

y_pred = dt.predict(X_test)

mse_dt = MSE(y_test, y_pred)

#root mse
rmse_dt = mse_dt**(1/2)

print(rmse_dt)
