from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
plt.style.use('ggplot')

digits = datasets.load_digits()

X = digits.data
y = digits.target

param_grid = {'n_neighbors':np.arange(1, 10)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, param_grid, cv=5) #Just like knn use this on diff classifiers like logreg, in the same manner

knn_cv.fit(X, y)

print(knn_cv.best_params_) #Shows best no. of n_neighbors suitable for this data

print(knn_cv.best_score_)
