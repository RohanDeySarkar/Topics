from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

iris = datasets.load_iris()

#print(type(iris))
#print(iris.keys(), iris.data, iris.target)
#print(iris.data.shape, iris.target_names) #data.shape shows rows,cols. Samples are in rows and features are in cols

X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)

#print(df.head)

#                              (dataframe, target variable, size of our figure, shape, marker_size)
_ = pd.plotting.scatter_matrix(df, c = y, figsize = [8,8], s=150, marker = 'D')

plt.show()
