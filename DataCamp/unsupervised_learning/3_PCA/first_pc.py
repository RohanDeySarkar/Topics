from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

iris = datasets.load_iris()

samples = iris.data
varieties = iris.target

#print(samples[1,:], samples[:,1], samples[14,1])

# Make a scatter plot of the untransformed points
plt.scatter(samples[:,0], samples[:,2])

model = PCA()

model.fit(samples)

mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()
