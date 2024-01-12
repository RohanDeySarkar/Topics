from sklearn import datasets
import matplotlib.pyplot as plt

##iris = datasets.load_iris()
##
##samples = iris.data[:,0] # for 1st col only
##varieties = iris.target

#print(iris.data.shape, iris.target_names) #data.shape shows rows,cols. Samples are in rows and features are in cols

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)   # consider setosa_petal_length and others  as predefined
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)

# Plot all ECDFs on the same plot
_ = plt.plot(x_set, y_set, marker='.', linestyle='none')
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')
_ = plt.plot(x_virg, y_virg, marker='.', linestyle='none')

# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.margins(0.02)
plt.show()

























