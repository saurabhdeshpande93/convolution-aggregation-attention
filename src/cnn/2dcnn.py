#Import all the required libraries

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from models import Models

## Ensure reproducibility
np.random.seed(123)
tf.random.set_seed(50)

## Load test-train datasets
srcpath = os.path.dirname(os.getcwd())
datapath = srcpath + "/frontiers_data/FEMData/"

X_train = np.load(datapath+'features_train_2D.npy')
Y_train = np.load(datapath+'labels_train_2D.npy')

X_test = np.load(datapath+'features_test_2D.npy')
Y_test = np.load(datapath+'labels_test_2D.npy')

## Define variables: Degrees of freedom, dimension, mesh resolution
dof, n_ch , n_x, n_y = 512, 2, 8, 32

## Converting inputs to the CNN compatible format.
n_train = len(X_train)
n_test = len(X_test)

X_train = X_train.reshape(n_train,n_ch, n_y, n_x)
Y_train = Y_train.reshape(n_train,n_ch, n_y, n_x)

X_test = X_test.reshape(n_test, n_ch, n_y, n_x)
Y_test = Y_test.reshape(n_test, n_ch, n_y, n_x)

## To preserve the shape as the real mesh
X_train = np.transpose(X_train, (0, 1, 3, 2))
Y_train = np.transpose(Y_train, (0, 1, 3, 2))
X_test = np.transpose(X_test, (0, 1, 3, 2))
Y_test= np.transpose(Y_test, (0, 1, 3, 2))

## Call/compile the model
input_shape = (n_ch, n_x, n_y)

UNET = Models().CNN(input_shape)

opt = Adam(lr = 0.0001)
UNET.compile(optimizer=opt, loss='mean_squared_error')


## Train the model
# Set training = true if you want to train from the scratch

training = False

if training == False:
    # Load the optimised parameters as used in the paper
    UNET.load_weights(srcpath+"/frontiers_data/trained_weights/2dcnn.h5")

else:
    # learning rate scheduler used in the paper
    def lr_scheduler(epoch, lr):

        if epoch <= 10:
            lr = lr
        elif 10 < epoch <= 200:
            lr = lr  - (0.0001 - 0.00001)/(190)
        else:
            lr = lr
        return lr

    # Callbacks for lr_scheduler and saving weights
    checkpoint = [LearningRateScheduler(lr_scheduler, verbose=1), ModelCheckpoint(srcpath+"/frontiers_data/trained_weights/2dcnnnew.h5", monitor='loss', verbose=1,
    save_best_only=True,save_weights_only=True, mode='min', period=1)]

    # Start the training procedure
    UNET.fit(X_train, Y_train,
                    epochs=32000,
                    batch_size=16,
                    shuffle=True,
                    validation_data=(X_test, Y_test),
                    callbacks=checkpoint)


## Predict solutions for the entire test set
predicts = UNET.predict(X_test)
predicts = np.transpose(predicts, (0, 1, 3, 2))
predicts= predicts.reshape((n_test,dof))
np.save(srcpath+"/frontiers_data/predictions/2dcnn_predicts.npy",predicts)
