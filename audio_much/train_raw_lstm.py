import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Dropout, concatenate, BatchNormalization, Activation, subtract, Lambda, Concatenate, AveragePooling2D, ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Reshape, Input, LSTM, RepeatVector, Convolution3D, Convolution1D, UpSampling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random as randy
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
from functools import partial
import gc
import h5py

encoder = LabelEncoder()
encoder.classes_ = np.load('songs_training_data_classes.npy')

timeseries_length = 200
mini_batch_size = 64


def build_lstm_audio_network(n_classes):
    input_shape = (timeseries_length, 33)
    inputs = Input(shape=input_shape)
    # lstm = LSTM(128, return_sequences=True)(inputs)
    # lstm = LSTM(32, return_sequences=False)(lstm)
    lstm = LSTM(200, return_sequences=False)(inputs)
    lstm = Dense(128, activation='relu')(lstm)
    lstm = Dense(n_classes, activation='softmax')(lstm)
    model = Model(inputs, lstm)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# build_lstm_audio_network(254).summary()


def mini_batch_generator(df_keys, training_data_filename):
    idx = 0
    while True:
        df_key = df_keys[idx]
        with h5py.File(training_data_filename, 'r') as hdf:
            x = hdf[df_key]['feature'][()]
            label = hdf[df_key]['label'][()]
        batch_size = x.shape[0] - timeseries_length
        mini_batch_count = int(np.ceil(batch_size / mini_batch_size))
        for mini_batch_idx in range(0, mini_batch_count):
            if mini_batch_idx == mini_batch_count - 1:
                curr_mini_batch_size = batch_size % mini_batch_size
            else:
                curr_mini_batch_size = mini_batch_size
            features = np.zeros((curr_mini_batch_size, timeseries_length, 33), dtype=np.float64)
            y = np_utils.to_categorical(encoder.transform([label] * curr_mini_batch_size), num_classes=encoder.classes_.shape[0])
            for batch_idx in range(0, curr_mini_batch_size):
                for timeseries_idx in range(0, timeseries_length):
                    features[batch_idx, timeseries_idx, :] = x[(mini_batch_idx * batch_idx) + timeseries_idx]
            yield features, y
        idx += 1
        if idx >= len(df_keys):
            idx = 0


def calculate_steps_epoch(df_keys, training_data_filename):
    batches = 0
    for df_key in df_keys:
        with h5py.File(training_data_filename, 'r') as hdf:
            x = hdf[df_key]['feature'][()]
        batch_size = x.shape[0] - timeseries_length
        batches += np.ceil(batch_size / mini_batch_size)
    return batches


training_data_filename = 'songs_training_data_22050_sequence_parts.h5'
validation_data_filename = 'songs_validation_data_22050_sequence_parts.h5'
lstm_model = build_lstm_audio_network(len(encoder.classes_))

# max_queue_size=1, workers=0, steps_per_epoch=1,
#with pd.HDFStore(training_data_filename) as hdf:
with h5py.File(training_data_filename, 'a') as hdf:
    hdf_keys = list(hdf.keys())
    lstm_model.fit_generator(mini_batch_generator(hdf_keys, training_data_filename),
                             validation_data=mini_batch_generator(hdf_keys, validation_data_filename),
                             validation_steps=calculate_steps_epoch(hdf_keys, training_data_filename),
                             steps_per_epoch=calculate_steps_epoch(hdf_keys, validation_data_filename),
                             epochs=10, use_multiprocessing=True)



lstm_model.save('model_raw_22050_lstm_01.h5')

from keras.models import load_model
lstm_model = load_model('model_raw_22050_lstm_01.h5')
