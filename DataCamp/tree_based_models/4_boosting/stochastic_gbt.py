from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split

SEED = 1

X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])

y = [0,1,0,1,0,1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

#max_features=0.2 -> each tree uses 20% of the available features
#subsamsple=0.8 -> each tree to sample 80% of the data for training
sbgt = GradientBoostingRegressor(max_depth=1, subsamsple=0.8, max_features=0.2, n_estimators=300, random_state=SEED)

sgbt.fit(X_train, y_train)

y_pred = sgbt.predict(X_test)

rmse_test = MSE(y_test, y_pred)**(1/2)

print('Test set RMSE:', rmse_test)
