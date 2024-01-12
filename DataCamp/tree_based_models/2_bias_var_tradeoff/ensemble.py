from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
# Import the VotingClassifier meta-model
from sklearn.ensemble import VotingClassifier
import numpy as np

SEED = 1

X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])

y = [0,1,0,1,0,1]

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Instantiate individual classifiers
lr = LogisticRegression(random_state=SEED)

knn = KNN(n_neighbors=2)

dt = DecisionTreeClassifier(random_state=SEED)

# Define a list called classifier that contains
# the tuples (classifier_name, classifier)
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]

for clf_name, clf in classifiers:
        
    clf.fit(X_train, y_train)             
    y_pred = clf.predict(X_test)
            
    print(clf_name,':', accuracy_score(y_test, y_pred))
                                       
vc = VotingClassifier(estimators=classifiers)

vc.fit(X_train, y_train)

y_pred = vc.predict(X_test)

print('Voting Classifier:', accuracy_score(y_test, y_pred))
















                 
                        
                        
