
from string import Template

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from data_prep import (
  read_dataset_from_csv,
  preprocess_data_and_get_X_and_y,
  partition_into_training_and_testing)

def grid_search_for_best_random_forest(parameters, x_train, y_train):
  base_model = RandomForestClassifier(random_state=42)

  return GridSearchCV(
    estimator=base_model,
    param_grid=parameters,
    return_train_score=True,
    n_jobs=4,
    cv=10,
    error_score='raise',
    verbose=1
  ).fit(x_train, y_train)


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


if __name__ == '__main__':
  import sys
  file_name = sys.argv[1]

  data_set = read_dataset_from_csv(file_name)
  X, y = preprocess_data_and_get_X_and_y(data_set)
  X_train, X_test, y_train, y_test = partition_into_training_and_testing(X, y, random_state=0)

  grid_search_parameters = {
    #'criterion':['gini', 'entropy'],
    'n_estimators':[50, 100, 150],
    'max_depth': [3, 5, 8],
  }

  grid_search_results = grid_search_for_best_random_forest(grid_search_parameters, X_train, y_train)
  best_estimator = grid_search_results.best_estimator_
  test_score = best_estimator.score(X_test, y_test)

  print_data_info(X.shape[0], X_test.shape[0], X_train.shape[0])
  print_statistics(grid_search_results, test_score)
