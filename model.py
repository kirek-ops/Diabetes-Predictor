import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.preprocessing import MinMaxScaler

# Import Data
data_path = 'data/diabetes.csv'
df = pd.read_csv(data_path)

y = df['Outcome']
X = df.drop('Outcome', axis = 1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 137)

model = Sequential([
    Flatten(input_shape = (8,)),
    Dense(8, activation = 'relu'),
    Dense(8, activation = 'relu'),
    Dense(6, activation = 'relu'),
    Dense(4, activation = 'relu'),
    Dense(2, activation = 'softmax'),
])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 5)

predictions = model.predict(x_test)
result = np.array([0 if x > y else 1 for x, y in predictions])

model.save('model.model')

loss = np.mean((result - y_test) ** 2)
print(f"Loss: {loss}")