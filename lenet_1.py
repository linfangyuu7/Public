'''
https://github.com/shap/shap/blob/master/notebooks/image_examples/image_classification/Front%20Page%20DeepExplainer%20MNIST%20Example.ipynb
'''
# this is the code from here --> https://github.com/keras-team/keras/blob/master/examples/demo_mnist_convnet.py
import os

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
# constants
WEIGHTS='LeNet_2.weights.h5'
batch_size = 256
kernel_size = (5, 5)

      # block1
      layers.Conv2D(4, kernel_size, activation='relu', padding='same', name='block1_conv1'
        , input_shape=input_shape), #need to specify the input shape in advance in order to load weights
      layers.MaxPooling2D(pool_size=(2, 2), name='block1_pool1'),
      # block2
      layers.Flatten(name='flatten'),

def architecture():
    #Define the model here
    model = keras.Sequential(
        [
          layers.Input(shape=input_shape),
          # block1
          layers.Conv2D(4, kernel_size, activation='relu', padding='same', name='block1_conv1'
            , input_shape=input_shape), #need to specify the input shape in advance in order to load weights
          layers.MaxPooling2D(pool_size=(2, 2), name='block1_pool1'),
          # block2
          layers.Flatten(name='flatten'),
          layers.Dense(num_classes)#, activation="softmax"),
        ]
    )
    return model
    
def new(x_train, y_train, h5=WEIGHTS, nb_epoch = 20, 
                      chk_pt = 'checkpoint', name4saving = 'epoch_{epoch:02d}.weights.h5', patience = 10):
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
    
    # training
    return train(model, x_train, y_train, h5, nb_epoch, chk_pt, name4saving, patience)

def train(model, x_train, y_train, h5, nb_epoch, chk_pt, name4saving = 'epoch_{epoch:02d}-val_loss-{val_loss:.4f}.weights.h5', patience = 10):
    # helper function for training a new model
    model = compile(model)
    
    # defining paths and callbacks
    from pathlib import PurePath
    dir4saving = PurePath('path2checkpoint', chk_pt)
    os.makedirs(dir4saving, exist_ok = True)

    filepath = os.path.join(dir4saving, name4saving)
    mcCallBack_loss = keras.callbacks.ModelCheckpoint(filepath, monitor = 'val_loss',
                                                verbose = 1, save_weights_only = True,)
    esCallBack = keras.callbacks.EarlyStopping(monitor = 'val_loss', verbose = 1, patience=patience)
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, validation_split=0.2,
              callbacks=[esCallBack, mcCallBack_loss])    
    # save model
    model.save_weights(h5)
    return model
def compile(model):
  # helper function for compiling a model
  model.compile( optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'], )
  return model

def load(h5):
    #load a model with the h5 weight file
    model = architecture()
    model.load_weights(h5)
    # compiling so that you may subsequently use the desired metrics for evaluation
    model = compile(model)
    return model
    
def evaluate(model, x, y):
    #show model characteristics
    model.summary()
    score = model.evaluate(x, y, verbose=0)
    print('\n')
    print('Overall Test loss:', score[0])
    print('Overall Test accuracy:', score[1])

def get_clean_data_set():
  # get training, validation and test sets
  print('Typically, an image format has one dimension for rows (height), one for columns (width) and one for channels.')
  print('However, we prepare another dimension to indicate how many samples. This is preferred by Tensorflow')
  print('The image data has been normalized to between -1 and +1')
  with open('y_train_new.npy', 'rb') as f:
    Y = np.load(f)
  with open('x_train_new.npy', 'rb') as f:
    X = np.load(f)
  return train_test_split(X, Y, test_size=0.32, random_state=42)

