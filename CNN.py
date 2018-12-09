import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

df_train =  pd.read_csv('train.csv')

df_train = df_train.drop(['key'],axis = 1)

df_train = df_train.dropna(inplace = False)

input_train_data = df_train.drop(['fare_amount'], axis = 1)
output_train_data = df_train['fare_amount']

def model(data,dim):
	W_1 = tf.Variable(tf.random_uniform([dim,10]))
	b_1 = tf.Variable(tf.zeros([10]))
	layer_1 = tf.add(tf.matmul(data,W_1), b_1)
	layer_1 = tf.nn.relu(layer_1)
	W_2 = tf.Variable(tf.random_uniform([10,10]))
	b_2 = tf.Variable(tf.zeros([10]))
	layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
	layer_2 = tf.nn.relu(layer_2)
	W_O = tf.Variable(tf.random_uniform([10,1]))
	b_O = tf.Variable(tf.zeros([1]))
	output = tf.add(tf.matmul(layer_2,W_O), b_O)
	return output
xs = tf.placeholder("float")
ys = tf.placeholder("float")
output =model(xs,len(input_train_data.columns))
cost = tf.reduce_mean(tf.square(output-ys))
train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    for i in range(100):
        for j in range(input_train_data.shape[0]):
            sess.run([cost,train],feed_dict=    {xs:input_train_data[j,:].reshape(1,3), ys:output_train_data[j]})
            # Run cost and train with each sample
        c_t.append(sess.run(cost, feed_dict={xs:input_train_data,ys:output_train_data}))
        # c_test.append(sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
        print('Epoch :',i,'Cost :',c_t[i])
    # pred = sess.run(output, feed_dict={xs:X_test})
    # predict output of test data after training
    # print('Cost :',sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
    # y_test = denormalize(df_test,y_test)
    # pred = denormalize(df_test,pred)

