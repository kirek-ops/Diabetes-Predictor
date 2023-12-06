import pandas as pd
import numpy as np

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Dropout, ReLU

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import joblib
from sklearn.linear_model import LogisticRegression

# Import Data
data_path = 'data/diabetes.csv'
df = pd.read_csv(data_path)

y = df['Outcome']
X = df.drop('Outcome', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = LogisticRegression()
model.fit(x_train, y_train)

pred = model.predict(x_test)

print(accuracy_score(y_test, pred))

joblib.dump(model, "model.joblib")