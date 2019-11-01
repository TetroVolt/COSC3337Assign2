import sys
import pandas as pd
import numpy as np
from string import Template
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.model_selection import GridSearchCV
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
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
  
def print_statistics(results: GridSearchCV, test_score):
  output_template = Template(
  """
  Grid Search parameter space for Random Forest = $grid_search_params
  best estimator parameters found = $best_params
  best estimator mean training score   = $mean_train_score
  best estimator mean validation score = $mean_validation_score
  best estimator test score            = $test_score
  """)

  train_results = results.cv_results_
  best_params = results.best_params_
  param_index = results.cv_results_['params'].index(results.best_params_)

  mean_train_score = results.cv_results_['mean_train_score'][param_index]
  mean_validation_score = results.cv_results_['mean_test_score'][param_index]

  print(output_template.substitute(
    grid_search_params=results.param_grid,
    best_params=best_params,
    mean_train_score=mean_train_score,
    mean_validation_score=mean_validation_score,
    test_score=test_score
  ))


def print_data_info(n_examples, n_test_examples, n_training_examples):
  output_template = Template(
  """
  Dataset characteristics:
  Number of examples in the dataset = $n_examples
  Number of examples reserved for test set = $n_test_examples
  Number of examples reserved for training via 10 fold CV = $n_training_examples
  Class Distribution Ratio (N : EI : IE) = 2 : 1 : 1
  N features : 60, all categorical (DNA base pairs in 60 base pair long sequence)
  """)
  print(output_template.substitute(
    n_examples=n_examples,
    n_test_examples=n_test_examples,
    n_training_examples=n_training_examples
  ))
  
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

def graph_results(dtree):
    dot_data = StringIO()
    export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    Image(graph.create_png())
    graph.write_png("result.png")

def split(x, y):
    return train_test_split(x, y, test_size = 0.20, random_state = 10)

def main(args):
    data = importdata()
    x, y = preprocess_data_and_get_X_and_y(data)
    x_train, x_test, y_train, y_test = split(x, y)

    parameters = {
            'max_depth':range(3,20),
            'min_samples_split':[3,5,25],
            'min_samples_leaf':[1,10,20],
            }
    clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4,cv=10, return_train_score=True)
    clf.fit(x_train,y_train)

    best_estimator = clf.best_estimator_
    test_score = best_estimator.score(x_test, y_test)

    print_data_info(x.shape[0], x_test.shape[0], x_train.shape[0])
    print_statistics(clf, test_score)
    graph_results(clf.best_estimator_)

if (__name__ == "__main__"):
    main(sys.argv)
