from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

iris = datasets.load_iris()

samples = iris.data
varieties = iris.target

model = PCA()

model.fit(samples)

transformed = model.transform(samples)

print(transformed)

# Assign 0th column of transformed: xs
xs = transformed[:,0]

# Assign 1st column(by choice) of transformed: ys
ys = transformed[:,2]

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()


