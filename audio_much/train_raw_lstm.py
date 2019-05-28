import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Dropout, concatenate, BatchNormalization, Activation, subtract, Lambda, Concatenate, AveragePooling2D, ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Reshape, Input, LSTM, RepeatVector, Convolution3D, Convolution1D, UpSampling2D, GRU
from keras import backend as K
from keras.callbacks import EarlyStopping
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
from keras.callbacks import Callback
import collections

encoder = LabelEncoder()
encoder.classes_ = np.load('songs_training_data_classes.npy')

timeseries_length = 16
mini_batch_size = 4


def build_lstm_audio_network(n_classes):
    input_shape = (timeseries_length, 12)
    inputs = Input(shape=input_shape)
    # lstm = LSTM(128, return_sequences=True)(inputs)
    # lstm = LSTM(32, return_sequences=False)(lstm)
    lstm = LSTM(128, dropout=0.15, recurrent_dropout=0.5, return_sequences=False, unroll=True, implementation=2)(inputs)
    lstm = Dense(n_classes, activation='softmax')(lstm)
    model = Model(inputs, lstm)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class FullSongValidation(Callback):
    def __init__(self, validation_data_filename, verbose=False):
        super(Callback, self).__init__()
        self.validation_data_filename = validation_data_filename
        self.validation_data = {}
        with h5py.File(self.validation_data_filename, 'r', libver='latest', swmr=True) as hdf:
            for hdf_key in hdf.keys():
                x = hdf[hdf_key]['feature'][()]
                label = hdf[hdf_key]['label'][()]
                batch_size = x.shape[0]
                features = np.zeros((batch_size, timeseries_length, 12), dtype=np.float64)
                # y = np_utils.to_categorical(encoder.transform([label] * batch_size), num_classes=encoder.classes_.shape[0])
                for batch_idx in range(0, batch_size):
                    for timeseries_idx in range(0, timeseries_length):
                        features_idx = batch_idx + timeseries_idx
                        if features_idx < x.shape[0]:
                            features[batch_idx, timeseries_idx, :] = x[features_idx]
                self.validation_data[label] = features
    def on_epoch_end(self, epoch, logs={}):
        validation_size = len(self.validation_data)
        correct_count = 0
        for label in self.validation_data:
            features = self.validation_data[label]
            res = self.model.predict(features)
            classes = [encoder.classes_[np.argmax(res[idx])] for idx in np.arange(res.shape[0])]
            counts = collections.Counter(classes)
            [(max_label, occurrences)] = counts.most_common(1)
            if label == max_label:
                correct_count += 1
            else:
                correct_guesses = counts[label]
                label_accuracy = correct_guesses / features.shape[0]
                strongest_guesses = counts[max_label]
                strongest_inaccuracy = strongest_guesses / features.shape[0]
                print("Incorrect song: {:s} - acc: {:.2f} - max inacc: {:.2f}".format(label, label_accuracy, strongest_inaccuracy))
        score = correct_count / validation_size
        print("Full song evaluation - epoch: {:d} - score: {:.2f}".format(epoch, score))


class MiniBatchGeneratorSequence(Sequence):
    def __init__(self, data_filename):
        self.data_filename = data_filename
        with h5py.File(self.data_filename, 'r', libver='latest', swmr=True) as hdf:
            self.hdf_keys = list(hdf.keys())
            self.batches = 0
            self.batch_metas = []
            for hdf_key in self.hdf_keys:
                x = hdf[hdf_key]['feature'][()]
                batch_size = x.shape[0]
                curr_batch_size = int(np.ceil(batch_size / mini_batch_size))
                self.batch_metas.append({'start': self.batches, 'end': self.batches + curr_batch_size, 'hdf_key': hdf_key})
                self.batches += curr_batch_size
    def __len__(self):
        return self.batches
    def __getitem__(self, overall_batch_idx):
        batch_meta = next(batch_meta for batch_meta in self.batch_metas
                          if batch_meta['start'] <= overall_batch_idx < batch_meta['end'])
        hdf_key = batch_meta['hdf_key']
        mini_batch_idx = overall_batch_idx - batch_meta['start']
        with h5py.File(self.data_filename, 'r', libver='latest', swmr=True) as hdf:
            x = hdf[hdf_key]['feature'][()]
            label = hdf[hdf_key]['label'][()]
        batch_size = x.shape[0]
        mini_batch_count = int(np.ceil(batch_size / mini_batch_size))
        if mini_batch_idx == mini_batch_count - 1 and batch_size % mini_batch_size > 0:
            curr_mini_batch_size = batch_size % mini_batch_size
        else:
            curr_mini_batch_size = mini_batch_size
        features = np.zeros((curr_mini_batch_size, timeseries_length, 12), dtype=np.float64)
        y = np_utils.to_categorical(encoder.transform([label] * curr_mini_batch_size),
                                    num_classes=encoder.classes_.shape[0])
        for batch_idx in range(0, curr_mini_batch_size):
            for timeseries_idx in range(0, timeseries_length):
                features_idx = (mini_batch_idx * batch_idx) + timeseries_idx
                if features_idx < x.shape[0]:
                    features[batch_idx, timeseries_idx, :] = x[features_idx]
        return features, y


training_data_filename = 'songs_training_data_44100_sequence_parts.h5'
validation_data_filename = 'songs_validation_data_44100_sequence_parts.h5'
full_song_validation_metric = FullSongValidation(validation_data_filename, verbose=True)
lstm_model = build_lstm_audio_network(len(encoder.classes_))
# earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

lstm_model.fit_generator(MiniBatchGeneratorSequence(training_data_filename),
                         validation_data=MiniBatchGeneratorSequence(validation_data_filename),
                         workers=3, max_queue_size=10, use_multiprocessing=True,
                         callbacks=[full_song_validation_metric],
                         epochs=100)


lstm_model.save('model_raw_22050_lstm_02.h5')

from keras.models import load_model
lstm_model = load_model('model_raw_22050_lstm_01.h5')
