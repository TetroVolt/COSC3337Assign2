#!/usr/bin/env python3

from typing import List
import sys

import pandas as pd
import sklearn as sk
import numpy as np

from sklearn.model_selection import train_test_split

def read_dataset_from_csv(file_name: str):
  return pd.read_csv(file_name)

def preprocess_data_and_get_X_and_y(data_set: pd.DataFrame):
  def one_hot_encoder(sequence_str):
    mapper = { # hardcoded for now
      'A': '10000000',
      'T': '01000000',
      'C': '00100000',
      'G': '00010000',
      'N': '00001000',
      'R': '00000100',
      'S': '00000010',
      'D': '00000001'
    }
    return list(''.join([mapper[base_pair] for base_pair in sequence_str]))

  features = np.array(list(map(one_hot_encoder, data_set['Sequence'])))
  labels = np.array(list(data_set['Class']))
  return features, labels

def partition_into_training_and_testing(X, y, random_state=0):
  return train_test_split(X, y, test_size=0.2, random_state=random_state)

def __MAIN__(args: List[str]):
  print(args)

if __name__ == '__main__':
  __MAIN__(sys.argv)
