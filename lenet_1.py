# -*- coding: utf-8 -*-

'''
LeNet-1 architecture
'''
#logits version
#For each sample, the model returns a vector of "logits" or "log-odds" scores, one for each class.

from __future__ import print_function
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Activation, Flatten
from keras.models import Model
import numpy as np
#import matplotlib.pyplot as plt
import datetime, os
# constants
WEIGHTS='LeNet_1.h5'
img_rows, img_cols = 28, 28
nb_classes = 10
kernel_size = (5, 5)
#The losses.SparseCategoricalCrossentropy loss takes a vector of logits and a True index and returns a scalar loss for each example.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def get_data_set():
  # get training, validation and test sets
  print('Typically, an image format has one dimension for rows (height), one for columns (width) and one for channels.')
  print('However, we prepare another dimension to indicate how many samples. This is preferred by Tensorflow')
  print('The image data has been normalized to between -1 and +1')
  with open('y_train.npy', 'rb') as f:
    y_train = np.load(f)
  with open('x_train.npy', 'rb') as f:
    x_train = np.load(f)
  with open('y_test.npy', 'rb') as f:
    y_test = np.load(f)
  with open('x_test.npy', 'rb') as f:
    x_test = np.load(f)
 # split test set into 2
  x, y = np.array_split(x_test, 2), np.array_split(y_test, 2)
  return (x_train, y_train), (x[0], y[0]), (x[1], y[1]) 

def get_clean_data_set():
  # get training, validation and test sets
  print('Typically, an image format has one dimension for rows (height), one for columns (width) and one for channels.')
  print('However, we prepare another dimension to indicate how many samples. This is preferred by Tensorflow')
  print('The image data has been normalized to between -1 and +1')
  with open('y_train_new.npy', 'rb') as f:
    y_train = np.load(f)
  with open('x_train_new.npy', 'rb') as f:
    x_train = np.load(f)
  return (x_train, y_train)
    
def architecture():
    #Define the model here
    input_shape = (img_rows, img_cols, 1)

    model = tf.keras.models.Sequential([
      # block1
      tf.keras.layers.Conv2D(4, kernel_size, activation='relu', padding='same', name='block1_conv1'
        , input_shape=input_shape), #need to specify the input shape in advance in order to load weights
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='block1_pool1'),
      # block2
      tf.keras.layers.Flatten(name='flatten'),
      tf.keras.layers.Dense(nb_classes, activation="softmax"),
    ])
    return model

def new(x_train, y_train, x_val, y_val, h5=WEIGHTS, nb_epoch = 20):
    #train a new model
    model = architecture()

    # Notice the pixel values are now in `[-1,1]`.
    print(np.min(x_train[:1]), np.max(x_train[:1]))

    ###############################Untrained model
    #For each example the model returns a vector of "logits" or "log-odds" scores, one for each class.
    predictions = model(x_train[:1]).numpy()
    print("Untrained model predictions (logits) for the 1st training example: ", predictions)  
    #The tf.nn.softmax function converts these logits to "probabilities" for each class:
    #Note: It is possible to bake this tf.nn.softmax in as the activation function for the last layer of the network. 
    # While this can make the model output more directly interpretable, this approach is discouraged as it's impossible
    # to provide an exact and numerically stable loss calculation for all models when using a softmax output.
    print("C.f. softmax: ", tf.nn.softmax(predictions).numpy())

    #The losses.SparseCategoricalCrossentropy loss takes a vector of logits and a True index and returns a scalar loss for each example.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #This loss is equal to the negative log probability of the the true class: It is zero if the model is sure of the correct class.
    #This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.log(1/10) ~= 2.3.
    print("Check loss (should be close to 2.3): ", loss_fn(y_train[:1], predictions).numpy())

    # compiling with your desired optimizer, loss function and metrics
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    # training
    return train(model, x_train, y_train, x_val, y_val, h5, nb_epoch)

def train(model, x_train, y_train, x_val, y_val, h5, nb_epoch):
    # helper function for training a new model
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=256, epochs=nb_epoch, 
      callbacks=[tensorboard_callback], verbose=1)
    
    # save model
    model.save_weights(h5)
    return model

def load(h5):
    #load a model with the h5 weight file
    model = architecture()
    model.load_weights(h5)
    print('LeNet-1 loaded')
    # compiling so that you may subsequently use the desired metrics for evaluation
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model
    
def evaluate(model, x, y):
    #show model characteristics
    model.summary()
    score = model.evaluate(x, y, verbose=0)
    print('\n')
    print('Overall Test loss:', score[0])
    print('Overall Test accuracy:', score[1])


def display_pattern(model, input_image, input_label, remarks=''):
  #display the image in Jupyter notebook  
  p = model(input_image).numpy() #convert from tensor to numpy
  prediction = np.argmax(p)

  a = np.squeeze(input_image)
  plt.figure()
  plt.imshow(a, cmap='gray', vmin=-1, vmax=1)
  plt.title('{}\n Ground truth: {} \n Predicted: {}'.format(remarks, input_label, prediction))
  plt.show()
