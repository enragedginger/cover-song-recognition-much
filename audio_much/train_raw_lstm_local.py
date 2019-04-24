import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Dropout, concatenate, BatchNormalization, Activation, subtract, Lambda, Concatenate, AveragePooling2D, ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Reshape, Input, LSTM, RepeatVector, Convolution3D, Convolution1D, UpSampling2D
from keras import backend as K
from keras.utils import np_utils, Sequence
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
import time

encoder = LabelEncoder()
encoder.classes_ = np.load('songs_training_data_classes.npy')

timeseries_length = 32
mini_batch_size = 16


def build_lstm_audio_network(n_classes):
    input_shape = (timeseries_length, 33)
    inputs = Input(shape=input_shape)
    # lstm = LSTM(128, return_sequences=True)(inputs)
    # lstm = LSTM(32, return_sequences=False)(lstm)
    lstm = LSTM(32, dropout=0.15, recurrent_dropout=0.35, return_sequences=False)(inputs)
    # lstm = Dense(100, activation='relu')(lstm)
    lstm = Dense(n_classes, activation='softmax')(lstm)
    model = Model(inputs, lstm)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# build_lstm_audio_network(254).summary()

class MiniBatchGeneratorSequence(Sequence):
    def __init__(self, hdf):
        self.hdf = hdf
        self.hdf_keys = list(hdf.keys())
        self.batches = 0
        self.batch_metas = []
        for hdf_key in self.hdf_keys:
            x = hdf[hdf_key]['feature'][()]
            batch_size = x.shape[0] - timeseries_length
            curr_batch_size = int(np.ceil(batch_size / mini_batch_size))
            self.batch_metas.append({'start': self.batches, 'end': self.batches + curr_batch_size, 'hdf_key': hdf_key})
            self.batches += curr_batch_size
    def __len__(self):
        return self.batches
    def __getitem__(self, mini_batch_idx):
        hdf_key = next(batch_meta['hdf_key']
                       for batch_meta in self.batch_metas if batch_meta['start'] <= mini_batch_idx < batch_meta['end'])
        x = hdf[hdf_key]['feature'][()]
        label = hdf[hdf_key]['label'][()]
        batch_size = x.shape[0] - timeseries_length
        mini_batch_count = int(np.ceil(batch_size / mini_batch_size))
        if mini_batch_idx == mini_batch_count - 1 and batch_size % mini_batch_size > 0:
            curr_mini_batch_size = batch_size % mini_batch_size
        else:
            curr_mini_batch_size = mini_batch_size
        features = np.zeros((curr_mini_batch_size, timeseries_length, 33), dtype=np.float64)
        y = np_utils.to_categorical(encoder.transform([label] * curr_mini_batch_size),
                                    num_classes=encoder.classes_.shape[0])
        for batch_idx in range(0, curr_mini_batch_size):
            for timeseries_idx in range(0, timeseries_length):
                features[batch_idx, timeseries_idx, :] = x[(mini_batch_idx * batch_idx) + timeseries_idx]
        return features, y


training_data_filename = 'songs_training_data_22050_sequence_parts.h5'
validation_data_filename = 'songs_validation_data_22050_sequence_parts.h5'
lstm_model = build_lstm_audio_network(len(encoder.classes_))


with h5py.File(training_data_filename, 'r') as hdf:
    with h5py.File(validation_data_filename, 'r') as validation_hdf:
        hdf_keys = list(hdf.keys())
        validation_hdf_keys = list(validation_hdf.keys())
        lstm_model.fit_generator(MiniBatchGeneratorSequence(hdf),
                                 validation_data=MiniBatchGeneratorSequence(validation_hdf),
                                 workers=3, max_queue_size=10, use_multiprocessing=False,
                                 epochs=10)


lstm_model.save('model_raw_22050_lstm_01.h5')

from keras.models import load_model
lstm_model = load_model('model_raw_22050_lstm_01.h5')