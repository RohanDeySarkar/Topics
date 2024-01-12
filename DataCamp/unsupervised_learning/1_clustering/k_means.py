from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = datasets.load_iris()

samples = iris.data

xs = samples[:,0] #sepal length is in 1st col 
ys = samples[:,2] #petal_length is in 3rd col

model = KMeans(n_clusters=3) #Since 3 targets
model.fit(samples)

print(model.inertia_) #use after fit

labels = model.predict(samples)

print(labels)

centroids = model.cluster_centers_

centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.scatter(xs, ys, c=labels)
plt.show()
