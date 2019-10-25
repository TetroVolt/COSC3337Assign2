#!/usr/bin/env python3

from typing import List
import sys

import pandas as pd
import sklearn as sk
import numpy as np

def read_dataset(file_name: str):
  return pd.read_csv(file_name)

def __MAIN_METHOD__(args: List[str]):
  print(args)

if __name__ == '__main__':
  __MAIN_METHOD__(sys.argv)
