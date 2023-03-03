import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

def filter_dataset(x, y, part):
    #Filter data of class 5 and 6.
    if part == 'Part A':
        keep = (y == 5) | (y == 6)
        x, y = x[keep], y[keep]
        y = y == 6
    else:
        keep = (y == 6)
        x, y = x[keep], y[keep]
        y = y == 6
        
    return x,y

def load_dataset(part):
    #Loads data from MNIST.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, y_train = filter_dataset(x_train, y_train, part)
    x_test, y_test = filter_dataset(x_test, y_test, part)
    
    return x_train, y_train, x_test, y_test

def split_dataset(x_train, y_train):
    #Spliting data to 80% train and 20% validation.
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

    return x_train, y_train, x_val, y_val

def reshape_data(x_train, x_val, x_test):
    #Reshapes 28x28 images to a single 1x784 vector.
    x_train = x_train.reshape(x_train.shape[0],784)
    x_val = x_val.reshape(x_val.shape[0],784)
    x_test = x_test.reshape(x_test.shape[0],784)

    return x_train, x_val, x_test

def normalization(x_train, x_test):
    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    return x_train, x_test
