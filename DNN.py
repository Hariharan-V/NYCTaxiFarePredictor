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



def neural_net_model(X_data,input_dim):
    W_1 = tf.Variable(tf.random_uniform([input_dim,10]))
    b_1 = tf.Variable(tf.zeros([10]))
    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)

    W_2 = tf.Variable(tf.random_uniform([10,10]))
    b_2 = tf.Variable(tf.zeros([10]))
    layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)
 
    W_O = tf.Variable(tf.random_uniform([10,1]))
    b_O = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer_2,W_O), b_O)
  
    return output

c_t = []
c_test = []
df_train = df_train.as_matrix()
df_train_output = df_train_output.as_matrix()
df_test = df_test.as_matrix()
df_test_output = df_test_output.as_matrix()
xs = tf.placeholder("float")
ys = tf.placeholder("float")
output = neural_net_model(xs,df_train.shape[1])
cost = tf.reduce_mean(tf.square(output-ys))
# mean squared error cost function
train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
with tf.Session() as sess:
  
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    for i in range(100):
        for j in range(df_train.shape[0]):
            sess.run([cost,train],feed_dict=    {xs:df_train[j,:].reshape(1,df_train.shape[1]), ys:df_train[j]})
            # Run cost and train with each sample
        c_t.append(sess.run(cost, feed_dict={xs:df_train,ys:df_train_output}))
        c_test.append(sess.run(cost, feed_dict={xs:df_test,ys:df_test_output}))
        print('Epoch :',i,'Cost :',c_t[i])
    pred = sess.run(output, feed_dict={xs:df_test})

    print('Cost :',sess.run(cost, feed_dict={xs:df_test,ys:df_test_output}))
