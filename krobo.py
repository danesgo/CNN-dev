import numpy as np
np.random.seed(123)

from keras.models import Sequetial
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
#from keras.utils import np_utils
#from keras.datasets import mnist

#Load data
#Preprocess input data
#pre process class labels

"""No se si cambiar los ZeroPadding2D por la opcion de same padding en las
capas Conv2D se supone es como lo mismo """

#Define model architecture
def vgg_16(weights_path=None):

    model = Sequetial()
    model.add(ZeroPadding2D((1,1), input_shape = (3,224,224)))
    model.add(Conv2D(64, 3, 3, activation='relu')) #Conv2D
    model.add(ZeroPadding2D((1,1))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation = 'relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation = 'relu'))
    model.add(MaxPooling2D((2,2), strides = (2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation = 'relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation = 'relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation = 'relu'))
    model.add(MaxPooling2D((2,2), strides = (2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation = 'relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation = 'relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation = 'relu'))
    model.add(MaxPooling2D((2,2), strides = (2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation = 'relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation = 'relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation = 'relu'))
    model.add(MaxPooling2D((2,2), strides = (2,2)))

    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

    
