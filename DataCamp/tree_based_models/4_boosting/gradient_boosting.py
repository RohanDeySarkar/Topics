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

gbt = GradientBoostingRegressor(n_estimators=10, max_depth=1, random_state=SEED)

gbt.fit(X_train, y_train)

y_pred = gbt.predict(X_test)

rmse_test = MSE(y_test, y_pred)**(1/2)

print('Test set RMSE:', rmse_test)
