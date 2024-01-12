import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

logreg = LogisticRegression()

X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])

y = [0,1,0,1,0,1]

#roc_auc can also be done using cross validation
cv_scores = cross_val_score(logreg, X, y, cv=2, scoring='roc_auc')
print(cv_scores)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

y_pred_prob = logreg.predict_proba(X_test)[:,1] #[:,1] -> choosing the 2nd col, predict_proba -> returns a probability
#y_pred_prob = logreg.predict_proba(X_test)[:,0] #returns 2 cols, here taking the  1st col

# false +ve rate, true +ve rate, threshold
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob) #roc_curve(actual labels, predicted probabilities),Compute Receiver operating characteristic(ROC)[only for binary classification task]

print(roc_auc_score(y_test, y_pred_prob))

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('Fale +ve rate')
plt.ylabel('True +ve rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
