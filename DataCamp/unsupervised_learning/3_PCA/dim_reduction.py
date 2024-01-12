from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

iris = datasets.load_iris()

samples = iris.data # 4 features
species = iris.target

pca = PCA(n_components=2) # 2 features -> n_components=2, 4 features reduced to 2
pca.fit(samples)
transformed = pca.transform(samples)

print(transformed.shape) # shows (150,2), 2 features

xs = transformed[:,0]
ys = transformed[:,1]

plt.scatter(xs, ys, c=species) # color represents 3 species of iris
plt.show()
