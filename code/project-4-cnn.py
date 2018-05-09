#!/bin/python3

# module imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from tensorflow.examples.tutorials.mnist import input_data
import math

# set working directory
os.chdir('/Users/jasonrich/msim607-machine-learning/project4/mnist-data/')

# download data from MNist
mnist = input_data.read_data_sets('../mnist-data',one_hot = True)


# helper functions

# initial weights
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(init_random_dist)

# initial bias
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape = shape)
    return tf.Variable(init_bias_vals)

# 2D convolution convenience function 
def conv2d(x,W):
    # x --> input tensor (batch of images, height(H), width (W), channels)
    # W --> kenneral [filter height, filter width, channels in, channels out]
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME') # SAME= 0's for padding

# pooling convenience function 
def max_pooling2x2(x):
    # x --> [batch, height, width, channel]
    return tf.nn.max_pool(x, ksize =[1,2,2,1] ,strides=[1,2,2,1], padding = 'SAME')

# convoltional layer
def conv_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

# normal fully connected layer
def norm_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b
    
### placeholders
    
## Note: 
# None = is the size of the batch
# 784 is the size of the pixels (28 x 28)
x = tf.placeholder(tf.float32, shape = [None, 784],name="x-in") 
y_true = tf.placeholder(tf.float32, shape = [None, 10], name="y-in") # 10 because they are one hot encoded
keep_prob = tf.placeholder("float")

### layers

# Note: 
# 28, 28 is height and width, and 1 (grey scale) is the channel
x_image = tf.reshape(x, [-1, 28, 28, 1])
hidden_1 = slim.conv2d(x_image,5,[5,5])
pool_1 = slim.max_pool2d(hidden_1,[2,2])
hidden_2 = slim.conv2d(pool_1,5,[5,5])
pool_2 = slim.max_pool2d(hidden_2,[2,2])
hidden_3 = slim.conv2d(pool_2,20,[5,5])
hidden_3 = slim.dropout(hidden_3,keep_prob)
out_y = slim.fully_connected(slim.flatten(hidden_3),10,activation_fn=tf.nn.softmax)

## Note:
# 5x5 conv layer that computes 32 feature per each 5x5 path
# 32 is the output channel
# 1 is the input channel
conv_1 = conv_layer(x_image, shape = [5,5,1,32])

conv_1_pooling = max_pooling2x2(conv_1)
conv_2 = conv_layer(conv_1_pooling, shape = [5,5,32,64]) # 64 features
conv_2_pooling = max_pooling2x2(conv_2)

# flatten results layer
conv_2_flat = tf.reshape(conv_2_pooling, [-1, 7*7*64])
full_layer_one = tf.nn.relu(norm_full_layer(conv_2_flat, 1024))

### dropout to prevent overfitting
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

# predictions
y_pred = norm_full_layer(full_one_dropout, 10)

# cross entropy lost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_pred))

# optimize using Adam Optimizer 
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train = optimizer.minimize(cross_entropy)

# initialize variables

init = tf.global_variables_initializer()
steps = 200
# steps = 2500
# steps = 5000
# steps = 10000

# Run the CNN 
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(50)
        #hold_prob:0.5 randomly dropout 50% of the neurons
        sess.run(train, feed_dict = {x:batch_x, y_true:batch_y,hold_prob:0.5})
        
        if i%100 == 0:
            print("ON STEP: {}".format(i))
            print("ACCURACY: ")
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1)) # eval model
            acc = tf.reduce_mean(tf.cast(matches,tf.float32)) # test data accuracy
            print(sess.run(acc, feed_dict={x:mnist.test.images, y_true:mnist.test.labels, hold_prob:1.0}))
            print('\n')

# visualize the network

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(tmp.shape[3] / n_columns)
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
        plt.show()


def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,784],order='F'),keep_prob:1.0})
    plotNNFilter(units)

        
imageToUse = mnist.test.images[0]
plt.imshow(np.reshape(imageToUse,[28,28]), interpolation="nearest", cmap="gray")


getActivations(hidden_1,imageToUse) # first layer
getActivations(hidden_2,imageToUse) # second layer
getActivations(hidden_3,imageToUse) # third layer


