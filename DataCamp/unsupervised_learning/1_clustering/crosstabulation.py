# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df[ 'varieties'])

# Display ct
print(ct)

'''
You have the array samples of grain samples, and a list varieties giving the grain variety for each sample
    varities              Canadian wheat  Kama wheat  Rosa wheat
    labels                                           
    0                       0           1          60
    1                      68           9           0
    2                       2          60          10

'''
