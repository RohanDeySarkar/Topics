'''
finding the articles most similar to the article about the footballer Cristiano Ronaldo.
The NMF features you obtained earlier are available as nmf_features, while titles is a list of the article titles.
'''
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF

nmf = NFM(n_components=6)
nmf_features = nmf.fit_transform(articles)

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']

# Compute the dot products: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())
