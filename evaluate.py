#!/usr/bin/env python3
import dslabs_functions as df
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# the dataset file name is a command line argument
if len(sys.argv) != 2:
    print("Usage: evaluate.py <dataset>")
    sys.exit(1)

# load the dataset
data = pd.read_csv(sys.argv[1])

# split the dataset into training and testing sets

