#Lasso for feature selection

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

boston = pd.read_csv('Boston.csv')

X = boston.drop('medv', axis=1).values
y = boston['medv'].values

names = boston.drop('medv', axis=1).columns

lasso = Lasso(alpha=0.1)
                                    #coef are very imp, lasso shrinks the coef of less imp features to 0 and keep the imp features.
lasso_coef = lasso.fit(X, y).coef_  #extracting the coef and adding it to the lasso_coef

#Same as above
##lasso.fit(X, y)
##lasso_coef = lasso.coef_

_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60) # give names on the x axis of resp. pts. Shows rm is the most imp feature
_ = plt.ylabel('Coefficients')
plt.show()
