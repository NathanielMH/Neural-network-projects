import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

# Reading the data
df = pd.read_csv('data.csv', names = column_names)
df.head()

# Checking for missing values
missing values = df.isna().sum()

# Normalizing the data
df = df.iloc[:,1:]
df_norm = (df-df.mean())/df.std()
df_norm.head()

# Convert labeled value
y_mean = df['price'].mean()
y_std = df['price'].std()

def convert_label_value(pred):
    return int(pred * y_std + y_mean)
  
# Creating training and test sets

# Selecting features and labels
x = df_norm.iloc[:, :6]
x.head()

y = df_norm.iloc[:, -1]
y.head()

x_arr = x.values
y_arr = y.values

# Train and test split
x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size = 0.05, random_state = 0)

# Creating a model
def get_model():
    model = Sequential([
        Dense(10,input_shape = (6,), activation = 'relu'),
        Dense(20, activation = 'relu'),
        Dense(5, activation = 'relu'),
        Dense(1)
    ])
    model.compile(
    loss = 'mse',
    optimizer = 'adam'
    )
    return model

# Model training
es_cb = EarlyStopping(monitor = 'val_loss', patience = 5)
model = get_model()
preds_on_untrained = model.predict(x_test)

history = model.fit(
    x_train, y_train,
    validation_data = (x_test, y_test),
    epochs = 100,
    callbacks = [es_cb]
)

# Plot training and validation loss
plot_loss(history)

# Plot raw predictions
 preds_on_trained = model.predict(x_test)
compare_predictions(preds_on_untrained, preds_on_trained, y_test)

# Plot price predictions
price_untrained = [convert_label_value(y) for y in preds_on_untrained]
price_trained = [convert_label_value(y) for y in preds_on_trained]
price_test = [convert_label_value(y) for y in y_test]

compare_predictions(price_untrained, price_trained, price_test)
