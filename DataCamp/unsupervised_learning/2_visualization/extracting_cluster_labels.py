# Perform the necessary imports
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import normalize
import pandas as pd
from scipy.cluster.hierarchy import fcluster

iris = datasets.load_iris()

samples = iris.data
varieties = iris.target

#Normalizing the samples
normalized_samples = normalize(samples)

# Calculate the linkage: mergings
mergings = linkage(normalized_samples, method='complete')

# Use fcluster to extract labels: labels
labels = fcluster(mergings, 6, criterion='distance') #fluster converts hierarchical clustering to flat clustering

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)
