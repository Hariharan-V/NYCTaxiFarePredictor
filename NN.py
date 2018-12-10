import tensorflow as tf
from tensorflow import keras

import pandas as pd
import math
import numpy as np
df_train =  pd.read_csv('train.csv').drop(['key'],axis = 1)#.drop(['pickup_datetime'],axis = 1)


def convert_time_sec(time):
    time = time.split()[1]
    time = time.split(":")
    return int(time[0])*3600+int(time[1])*60+int(time[2])

df_train['pickup_datetime'] = df_train['pickup_datetime'].apply(convert_time_sec)

# print(df_train.mean())
df_train = df_train.sample(n = 2000,random_state=np.random.randint(200)).reset_index(drop=True)

df_test = df_train.sample(n=1000,random_state=np.random.randint( 200))
df_train = df_train.drop(df_test.index)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


df_train_output = df_train['fare_amount']
df_train = df_train.drop(['fare_amount'],axis = 1)

df_test_output = df_test['fare_amount']
df_test = df_test.drop(['fare_amount'],axis = 1)

mean = df_train.mean()
std = df_train.std()


df_train = (df_train - mean) / std
df_test = (df_test - mean) / std



def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(df_train.shape[1],)),
     keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])
  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print("."),
    


EPOCHS = 500
history = model.fit(df_train, df_train_output, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])
[loss, mae] = model.evaluate(df_train, df_train_output, verbose=0)
print("")
print("Training set Mean Abs Error:" + str(mae))

test_predictions = model.predict(df_test).flatten()
error = test_predictions - df_test_output

print("The neural network has a absolute mean error of "+str(np.absolute(error).mean(0)))