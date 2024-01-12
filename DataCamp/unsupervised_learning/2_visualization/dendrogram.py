# Perform the necessary imports
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import normalize

iris = datasets.load_iris()

samples = iris.data
varieties = iris.target

#Normalizing the samples
normalized_samples = normalize(samples)

# Calculate the linkage: mergings
mergings = linkage(normalized_samples, method='complete')
#mergings = linkage(normalized_samples, method='single')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
           )

plt.show()
