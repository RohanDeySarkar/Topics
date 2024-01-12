from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score
import numpy as np

X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])

y = [0,1,0,1,0,1]

# Set seed for reproducibility
SEED = 123

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
                                                         
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.14, random_state=SEED)

# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV = - cross_val_score(dt, X_train, y_train,  cv = 10, scoring='neg_mean_squared_error', n_jobs = -1)

dt.fit(X_train, y_train)

y_predict_train = dt.predict(X_train)

y_predict_test = dt.predict(X_test)

print('CV MSE: {:.2f}'.format(MSE_CV.mean()))

print('Train MSE: {:.2f}'.format(MSE(y_train, y_predict_train)))

print('Test MSE: {:.2f}'.format(MSE(y_test, y_predict_test)))
