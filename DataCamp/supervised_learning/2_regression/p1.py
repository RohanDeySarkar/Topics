import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from sklearn import linear_model

reg = linear_model.LinearRegression()

boston = pd.read_csv('Boston.csv')
#print(boston.head())

X = boston.drop('medv', axis=1).values

y = boston['medv'].values

X_rooms = X[:,5] #slicing the room col, we will only use 1 feature in this ex

#print(type(X_rooms), type(y))

y = y.reshape(-1, 1)
X_rooms = X_rooms.reshape(-1, 1)

reg.fit(X_rooms, y) #here taking only 1 feature, X_rooms

prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1,1) #taking random pts bet min and max of X_rooms and reshaping them

y_pred = reg.predict(prediction_space)

plt.scatter(X_rooms, y)
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.ylabel('Value of house /1000 ($)')
plt.xlabel('Number of rooms')
plt.show()
