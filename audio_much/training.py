import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
df = pd.read_pickle('songs_training_data.pick')

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
X = np.array(df['feature'].tolist())
y = np.array(df['label'].tolist())
lb = LabelEncoder()
dummy_y = np_utils.to_categorical(lb.fit_transform(y))

np.save('songs_training_data_classes.npy', lb.classes_)


from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics


def train_original_model():
    num_labels = dummy_y.shape[1]
    # build model
    model = Sequential()
    model.add(Dense(256, input_shape=(40,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.fit(X, dummy_y, batch_size=32, epochs=100)
    model.save('basic_af_model.h5')


def triplet_loss_fn(distance_fn, margin, anchor, positive, negative):
    max(distance_fn(anchor, positive) - distance_fn(anchor, negative) + margin, 0)


# At the heart of our music recognizer is the neural network fingerprinter (NNFP) which analyzes a few
# seconds of audio and emits a single fingerprint embedding at a rate of one per second. A detailed
# structure of the NNFP can be seen in Figure 2a. A stack of convolutional layers is followed by a
# two-level divide-and-encode block [10] which splits the intermediate representation into multiple
# branches. All layers except for the final divide-and-encode layer use the ELU [5] activation function
# and batch normalization.

model = Sequential()

model.add(Convolution2D(filters, kernel_size, strides=(1, 3)))
model.add(Convolution2D(filters, kernel_size, strides=(3, 1)))
model.add(Convolution2D(filters, kernel_size, strides=(1, 3)))
model.add(Convolution2D(filters, kernel_size, strides=(3, 1)))