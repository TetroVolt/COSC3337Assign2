from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from string import Template
import sys
from data_prep import (
  read_dataset_from_csv,
  preprocess_data_and_get_X_and_y,
  partition_into_training_and_testing)
import datetime

# Takes nparray with shape (n,3) produced by partition_into_training_and_testing
def reformatTargetForSVC(target: np.ndarray):
    y_format = []
    for item in target:
        if(item[0] == 1):
            y_format.append(0)
        elif(item[1] == 1):
            y_format.append(1)
        elif(item[2] == 1):
            y_format.append(2)
        else:
            print("FORMAT ERROR")
            sys.exit(1)
    return y_format

# Create a GridSearch with specified parameters and fit to training set
def gridSearchSVC(params, x_train, y_train):
    return GridSearchCV(estimator=SVC(random_state=0), 
                        param_grid=params,
                        return_train_score=True,
                        error_score='raise',
                        n_jobs=4,
                        cv=10,
                        verbose=1
            ).fit(X_train, y_train)

# Prints the info on the dataset
def print_data_info(n_examples, n_test_examples, n_training_examples):
  from string import Template
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

# Print the statistics of the GridSearch
def printStatistics(results: GridSearchCV, testScore):
    output_template = Template(
    """
    Grid Search parameter space for SVM = $grid_search_params
    best estimator parameters found = $best_params
    best estimator mean training score   = $mean_train_score +/- $std_train_score
    best estimator mean validation score = $mean_validation_score +/- $std_validation_score
    best estimator test score            = $test_score
    """)
    
    param_index = results.cv_results_['params'].index(results.best_params_)
    mean_train_score = results.cv_results_['mean_train_score'][param_index]
    std_train_score = results.cv_results_['std_train_score'][param_index]
    mean_validation_score = results.cv_results_['mean_test_score'][param_index]
    std_validation_score = results.cv_results_['std_test_score'][param_index]

    print(output_template.substitute(
        grid_search_params = results.param_grid,
        best_params = results.best_params_,
        mean_train_score = mean_train_score,
        std_train_score = std_train_score,
        mean_validation_score = mean_validation_score,
        std_validation_score = std_validation_score,
        test_score = testScore
    ))

newDataset = read_dataset_from_csv('..\\datasetSPLICEDNA\\splice.data.csv')
X, y = preprocess_data_and_get_X_and_y(newDataset)
# Split data into training and testing sets. CHANGE test_size TO 0.2/0.5/0.7 TO GET THE OTHER RESULTS
X_train, X_test, y_train, y_test = partition_into_training_and_testing(X, y, test_size=0.2)

y_train_format = reformatTargetForSVC(y_train)
y_test_format = reformatTargetForSVC(y_test)

# Parameters to modify and tune to test for a better model
param_grid = [
  {'C': [100, 1000, 10000], 
   'gamma': [0.01, 0.001], 
   'kernel': ['poly','rbf'], 
   'degree':[2,3,4]}
]

startTime = datetime.datetime.now()
gridSearchResults = gridSearchSVC(param_grid, X_train, y_train_format)
processTime = datetime.datetime.now() - startTime

# Get the best estimator from the GridSearchCV
best_est = gridSearchResults.best_estimator_
trainingScore = best_est.score(X_train, y_train_format)
testingScore = best_est.score(X_test, y_test_format)

printStatistics(gridSearchResults, testingScore)
print_data_info(X.shape[0], X_test.shape[0], X_train.shape[0])
print("Process Time\t\t= " + str(processTime/60) + " Minutes")
