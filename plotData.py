import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(__file__))
df = pd.read_csv("nonlinear_data.csv")
df = df.dropna()  # dropping small percentage of rows have missing values
X_train = df.iloc[:, 0:2].values  # Features (columns 1-2)
y_train = df['label'].values  # Target variable (column 3)
print("X_train", X_train)