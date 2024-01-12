# Import TSNE
from sklearn.manifold import TSNE
from sklearn import datasets
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

iris = datasets.load_iris()

samples = iris.data
varieties = iris.target

#Normalizing the samples
normalized_samples = normalize(samples)

# Create a TSNE instance: model
model = TSNE(learning_rate=100)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(normalized_samples)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c=varieties)
plt.show()
