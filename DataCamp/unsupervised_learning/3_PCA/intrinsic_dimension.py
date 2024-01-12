from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

iris = datasets.load_iris()

samples = iris.data
#features = iris.target

scaler = StandardScaler()
model = PCA()

pipeline = make_pipeline(scaler, model)
model.fit(samples)

features = range(model.n_components_) #each row defines displacement from mean -> n_components_

plt.bar(features, model.explained_variance_)
plt.ylabel('Variance')
plt.xlabel('PCA feature')
plt.xticks(features)
plt.show()

