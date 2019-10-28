
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from data_prep import (
  read_dataset_from_csv,
  preprocess_data_and_get_X_and_y,
  partition_into_training_and_testing)

def grid_search_for_best_random_forest(x_train, y_train):
  base_model = RandomForestClassifier()
  parameters = {
    'n_estimators':[10, 20, 30, 40],
    'criterion':['gini', 'entropy'],
    'max_depth':[1, 2, 3, 4, "None"],
    'max_leaf_nodes': [2, 4, 8, 16],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 10, 30, 50],
    'bootstrap': [True, False],
    'random_state': [42]
  }

  grid_search_results = GridSearchCV(
    estimator=base_model,
    param_grid=parameters,
    return_train_score=True,
    n_jobs=4,
    cv=10,
    error_score='raise'
  ).fit(x_train, y_train)

  return grid_search_results

def __MAIN__(file_name):
  data_set = read_dataset_from_csv(file_name)

  X, y = preprocess_data_and_get_X_and_y(data_set)

  X_train, X_test, y_train, y_test = partition_into_training_and_testing(X, y, random_state=0)
  gridSearchObject = grid_search_for_best_random_forest(X_train, y_train)
  print(gridSearchObject)

if __name__ == '__main__':
  import sys
  __MAIN__(sys.argv[1])
