#!/usr/bin/env python3

from typing import List
import sys

import pandas as pd
import sklearn as sk
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from string import Template

def print_statistics(results, test_score):
  # results is the object returned by GridSearchCV

  output_template = Template(
  """
  Grid Search parameter space for Random Forest = $grid_search_params
  best estimator parameters found = $best_params
  best estimator mean training score   = $mean_train_score +/- $std_train_score
  best estimator mean validation score = $mean_validation_score +/- $std_validation_score
  best estimator test score            = $test_score
  """)

  train_results = results.cv_results_
  best_params = results.best_params_
  param_index = results.cv_results_['params'].index(results.best_params_)

  mean_train_score = results.cv_results_['mean_train_score'][param_index]
  mean_validation_score = results.cv_results_['mean_test_score'][param_index]
  std_train_score = results.cv_results_['std_train_score'][param_index]
  std_validation_score = results.cv_results_['std_test_score'][param_index]

  print(output_template.substitute(
    grid_search_params=results.param_grid,
    best_params=best_params,
    mean_train_score=mean_train_score,
    mean_validation_score=mean_validation_score,
    std_train_score=std_train_score,
    std_validation_score=std_validation_score,
    test_score=test_score
  ))

def read_dataset_from_csv(file_name: str):
  return pd.read_csv(file_name)

def preprocess_data_and_get_X_and_y(data_set: pd.DataFrame):
  features = np.array(list(map(list, data_set['Sequence'])), dtype='str')
  labels = np.array(list(data_set['Class']), dtype='str').reshape(-1,1)

  base_pairs = ('A', 'C', 'D', 'G', 'N', 'R', 'S', 'T')
  categories = [base_pairs for i in range(features.shape[1])]

  features_encoder = OneHotEncoder(categories=categories).fit(features)
  labels_encoder = OneHotEncoder(handle_unknown='ignore').fit(labels)

  features = features_encoder.transform(features).toarray()
  labels = labels_encoder.transform(labels).toarray()

  return features, labels

def partition_into_training_and_testing(X, y, random_state=0, test_size=0.2):
  return train_test_split(X, y, test_size=test_size, random_state=random_state)

def __MAIN__(args: List[str]):
  print(args)

if __name__ == '__main__':
  __MAIN__(sys.argv)
