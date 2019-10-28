#!/usr/bin/env python3

from typing import List
import sys

import pandas as pd
import sklearn as sk
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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

def partition_into_training_and_testing(X, y, random_state=0):
  return train_test_split(X, y, test_size=0.2, random_state=random_state)

def __MAIN__(args: List[str]):
  print(args)

if __name__ == '__main__':
  __MAIN__(sys.argv)
