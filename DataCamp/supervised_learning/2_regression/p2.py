import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score

ridge = Ridge(alpha=0.1, normalize=True)
lasso = Lasso(alpha=0.1, normalize=True)
reg = LinearRegression()

boston = pd.read_csv('Boston.csv')
#print(boston.head())

X = boston.drop('medv', axis=1).values

y = boston['medv'].values

cv_results = cross_val_score(reg, X, y, cv=5) #cross_val_score evaluates a score by cross_validation
print(cv_results)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

lasso.fit(X_train, y_train)
lasso_pred = ridge.predict(X_test)

print(lasso.score(X_test, y_test))
print(ridge.score(X_test, y_test))
print(reg.score(X_test, y_test))

























