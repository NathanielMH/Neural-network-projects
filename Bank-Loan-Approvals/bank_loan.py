import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 

# Load dataset
bank_df = pd.read_csv('UniversalBank.csv')

# See how many null values exist in the dataframe
bank_df.isnull().sum()

# Prepare the data for training the model
# Specify model input features (all data except for the target variable) 
X = bank_df.drop(columns = ['Personal Loan'])

# Model output (target variable)
y = bank_df['Personal Loan']
y = to_categorical(y)

# Scale the data before training the model
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)

# Spliting the data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1)

# Create the model
# Create keras sequential model
ANN_model = keras.Sequential()

# Add dense layer
ANN_model.add(Dense(250, input_dim = 13, kernel_initializer = 'normal',activation = 'relu'))

ANN_model.add(Dropout(0.3))
ANN_model.add(Dense(500, activation = 'relu'))

ANN_model.add(Dropout(0.3))
ANN_model.add(Dense(500, activation = 'relu'))

ANN_model.add(Dropout(0.3))
ANN_model.add(Dense(500, activation = 'relu'))

ANN_model.add(Dropout(0.4))
ANN_model.add(Dense(250, activation = 'linear'))

ANN_model.add(Dropout(0.4))

# Add dense layer with softmax activation
ANN_model.add(Dense(2, activation = 'softmax'))

# Compile the model
ANN_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Train the model
history = ANN_model.fit(X_train, y_train, epochs = 20, validation_split = 0.2, verbose = 1)

# Evaluating model performance
# Make predictions
predictions = ANN_model.predict(X_test)

# Append the index of max value using argmax function
predict = []
for i in predictions:
    predict.append(np.argmax(i))

# Get the acccuracy of the model
result = ANN_model.evaluate(X_test, y_test)

print("Accuracy : {}".format(result[1]))

# Print out the classification report
print(classification_report(y_original, predict))
