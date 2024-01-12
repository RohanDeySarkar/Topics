from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np

SEED = 1

X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])

y = [0,1,0,1,0,1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)

bc = BaggingClassifier(base_estimator = dt, n_estimators=10, oob_score=True, n_jobs=-1)  # n_estimators -> here it is 10 dt(base_estimator)

bc.fit(X_train, y_train)

y_pred = bc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Bagging Classifier:', accuracy)

oob_accuracy = bc.oob_score_
print(oob_accuracy)
