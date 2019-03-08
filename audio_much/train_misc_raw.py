import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
from keras.models import Sequential, Model
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D
from kapre.augmentation import AdditiveNoise
from kapre.filterbank import Filterbank
from keras.layers import Lambda, Concatenate, AveragePooling2D, ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Reshape, Input, LSTM, RepeatVector, Convolution3D, Convolution1D, UpSampling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
from random import shuffle

from audio_much.core import build_label_encoder

encoder = LabelEncoder()
encoder.classes_ = np.load('songs_training_data_classes.npy')

def build_model(total_classes):
    # 6 channels (!), maybe 1-sec audio signal, for an example.
    sample_rate = 22050
    input_shape = (1, sample_rate)
    model = Sequential()
    # A mel-spectrogram layer
    model.add(Melspectrogram(n_dft=512, n_hop=256, input_shape=input_shape,
                             padding='same', sr=sample_rate, n_mels=128,
                             fmin=0.0, fmax=sample_rate/2, power_melgram=1.0,
                             return_decibel_melgram=False,
                             trainable_fb=False,
                             trainable_kernel=False))
    # model.add(Melspectrogram(512, input_shape=input_shape))
    # Maybe some additive white noise.
    # model.add(AdditiveNoise(power=0.2))
    # If you wanna normalise it per-frequency
    model.add(Normalization2D(str_axis='freq'))  # or 'freq', 'channel', 'time', 'batch', 'data_sample'
    model.add(Convolution2D(32, (3, 3), name='conv1', activation='relu'))
    model.add(MaxPooling2D((25, 17)))
    model.add(Convolution2D(32, (2, 2), name='conv2', activation='relu'))
    model.add(Flatten())
    model.add(Dense(total_classes, activation='softmax'))
    # model.add(Reshape((128, 87, 1, 664)))
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy']) # if single-label classification
    # and train it
    # model.fit(x, y)
    # model.save('melspectorgram_raw_22050_model.h5')
    return model


def build_autoencoder_models(total_classes):
    sample_rate = 22050
    input_shape = (1, sample_rate)
    filterbank_n_fbs = 50
    timesteps = filterbank_n_fbs
    input_dim = 87
    latent_dim = 74
    inputs = Input(shape=input_shape)
    spectrogram = Spectrogram(n_dft=512, n_hop=256, input_shape=input_shape,
                          return_decibel_spectrogram=True, power_spectrogram=2.0,
                          trainable_kernel=False, name='static_stft')(inputs)
    filterbank = Filterbank(n_fbs=filterbank_n_fbs, trainable_fb=False, sr=sample_rate, init='mel', fmin=0, fmax=sample_rate // 2, bins_per_octave=12,
                         name='mel_bank')(spectrogram)
    reshape = Reshape((filterbank_n_fbs, -1))(filterbank)
    # flatten = Flatten()(filterbank)
    # dense = Dense(sample_rate)(flatten)
    encoded = LSTM(int(input_dim / 2),
                   activation="relu",
                   return_sequences=True)(reshape)
    encoded = LSTM(latent_dim,
                   activation="relu",
                   return_sequences=False)(encoded)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(int(input_dim / 2),
                   activation="relu",
                   return_sequences=True)(decoded)
    decoded = LSTM(input_dim,
                   return_sequences=True)(decoded)
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    spectro_filter_bank = Model(inputs, reshape)
    autoencoder.compile(optimizer='adagrad',
                        loss='mse',
                        metrics=['acc'])
    return autoencoder, encoder, spectro_filter_bank


def build_conv_divide_model(total_classes):
    sample_rate = 22050
    input_shape = (1, sample_rate)
    filterbank_n_fbs = 64
    inputs = Input(shape=input_shape)
    # spectrogram = Spectrogram(n_dft=1024, input_shape=input_shape,
    #                       return_decibel_spectrogram=True, power_spectrogram=1.0,
    #                       trainable_kernel=False, name='static_stft')(inputs)
    spectrogram = Melspectrogram(sr=sample_rate, n_dft=1024, input_shape=input_shape,
                                 return_decibel_melgram=True, trainable_kernel=False)(inputs)
    # filterbank = Filterbank(n_fbs=filterbank_n_fbs, trainable_fb=False, sr=sample_rate, init='mel', fmin=0,
    #                         fmax=sample_rate // 2, bins_per_octave=12,
    #                         name='mel_bank')(spectrogram)
    # reshape = Reshape((filterbank_n_fbs, -1))(filterbank)
    reshape = ZeroPadding2D(padding=((1, 7), (1, 3)), data_format=None)(spectrogram)
    # flatten = Flatten()(filterbank)
    # dense = Dense(sample_rate)(flatten)
    encoded = Convolution2D(64, (1, 3), activation='relu', padding='same')(reshape)
    encoded = Convolution2D(32, (1, 3), activation='relu', padding='same')(encoded)
    # encoded = MaxPooling2D((2, 2), padding='same')(encoded)
    encoded = Convolution2D(16, (3, 1), activation='relu', padding='same')(encoded)
    # encoded = MaxPooling2D((2, 2), padding='same')(encoded)
    encoded = Convolution2D(8, (1, 3), activation='relu', padding='same')(encoded)
    # encoded = MaxPooling2D((2, 2), padding='same')(encoded)
    decoded = Convolution2D(8, (1, 3), activation='relu', padding='same')(encoded)
    # decoded = UpSampling2D((2, 2))(decoded)
    decoded = Convolution2D(16, (3, 1), activation='relu', padding='same')(decoded)
    # decoded = UpSampling2D((2, 2))(decoded)
    decoded = Convolution2D(32, (1, 3), activation='relu', padding='same')(decoded)
    # decoded = UpSampling2D((2, 2))(decoded)
    decoded = Convolution2D(64, (1, 3), activation='relu', padding='same')(decoded)
    # decoded = UpSampling2D((2, 2))(decoded)
    decoded = Convolution2D(1, (2, 2), activation='sigmoid', padding='same')(decoded)
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    spectro_filter_bank = Model(inputs, reshape)
    autoencoder.compile(optimizer='adam',
                        loss='mse',
                        metrics=['acc'])
    return autoencoder, encoder, spectro_filter_bank



autoencoder, encoder_model, spectro_filter_bank = build_conv_divide_model(12)
autoencoder.summary()

with pd.HDFStore('songs_training_data_22050_raw_parts.h5') as hdf:
    for df_key in hdf.keys():
        print('Reading and training df: ' + df_key)
        # df = pd.read_hdf('songs_training_data_22050_raw_parts.h5', key=df_key)
        df = hdf.get(df_key)
        X = np.array(df['feature'].tolist())
        reshaped_x = X[:, np.newaxis, :]
        y = spectro_filter_bank.predict(reshaped_x)
        autoencoder.fit(reshaped_x, y, epochs=1)

model = build_model(encoder.classes_.shape[0])

with pd.HDFStore('songs_training_data_22050_raw_parts.h5') as hdf:
    for df_key in hdf.keys():
        print('Reading and training df: ' + df_key)
        # df = pd.read_hdf('songs_training_data_22050_raw_parts.h5', key=df_key)
        df = hdf.get(df_key)
        y = np_utils.to_categorical(encoder.transform(df['label'].tolist()), num_classes=encoder.classes_.shape[0])
        X = np.array(df['feature'].tolist())
        reshaped_x = X[:, np.newaxis, :]
        model.fit(reshaped_x, y, epochs=10)


model.save('melspectorgram_raw_22050_model.h5')




input_shape = (1, 22050)
sr = 22050
model = Sequential()
# A mel-spectrogram layer
model.add(Melspectrogram(n_dft=512, n_hop=256, input_shape=input_shape,
                         padding='same', sr=sr, n_mels=128,
                         fmin=0.0, fmax=sr/2, power_melgram=1.0,
                         return_decibel_melgram=False, trainable_fb=False,
                         trainable_kernel=False,
                         name='trainable_stft'))
# Maybe some additive white noise.
model.add(AdditiveNoise(power=0.2))
# If you wanna normalise it per-frequency
model.add(Normalization2D(str_axis='freq')) # or 'channel', 'time', 'batch', 'data_sample'
# After this, it's just a usual keras workflow. For example..
# Add some layers, e.g., model.add(some convolution layers..)
# Compile the model
model.compile('adam', 'categorical_crossentropy') # if single-label classification
model.fit(reshaped_x, dummy_y)