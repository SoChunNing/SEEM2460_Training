import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import sys

script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
data = pd.read_csv(f'{script_dir}/stargazing_data.csv')
