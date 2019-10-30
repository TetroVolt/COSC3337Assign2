import sys
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.model_selection import GridSearchCV

from data_prep import preprocess_data_and_get_X_and_y
def importdata():
    data = pd.read_csv('../datasetSPLICEDNA/splice.data.csv')
    data = data.drop('Source', axis = 1)

    return data

def train(X_train, X_test, y_train, method): 
    clf = DecisionTreeClassifier( criterion = method,
            random_state = 100, max_depth = 3, min_samples_leaf = 5) 
  
    clf.fit(X_train, y_train) 
    return clf
  
"""  
def prediction(X_test, clf_object): 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
def check_accuracy(y_test, y_pred): 
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", accuracy_score(y_test,y_pred)*100) 
    print("Classification Report : ", classification_report(y_test, y_pred)) 
"""

def split(x, y):
    return train_test_split(x, y, test_size = 0.20, random_state = 10)

def main(args):
    data = importdata()
    x, y = preprocess_data_and_get_X_and_y(data)
    x_train, x_test, y_train, y_test = split(x, y)

    parameters = {'max_depth':range(3,20)}
    clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4,cv=10)
    clf.fit(x_train,y_train)

    best_estimator = clf.best_estimator_
    print(best_estimator)
    print(clf.best_score_, clf.best_params_)

if (__name__ == "__main__"):
    main(sys.argv)
