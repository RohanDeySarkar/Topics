# Import necessary modules
#data is give as X:target, y:features[This program will not run as X and y in undefined(just assume them as features and labels)]
#tuning for diff vals of alpha

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import numpy as np

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(alpha=0.1, normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = Ridge(alpha=alpha, normalize=True)
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge.alpha, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)

'''
Selects Most appropiate model for the given data(Tuning alpha value) -> Ridge
Selects Most imp feature -> Lasso
'''
