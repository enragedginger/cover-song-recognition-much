import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
from keras.models import Sequential, Model
from keras.optimizers import Adam
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from kapre.augmentation import AdditiveNoise
from keras.layers import Dropout, concatenate, BatchNormalization, Activation, subtract, Lambda, Concatenate, AveragePooling2D, ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Reshape, Input, LSTM, RepeatVector, Convolution3D, Convolution1D, UpSampling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
from random import shuffle
import tensorflow as tf
from functools import partial

from audio_much.core import build_label_encoder

encoder = LabelEncoder()
encoder.classes_ = np.load('songs_training_data_classes.npy')

#src: https://github.com/keras-team/keras/issues/9498
def best_triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
    return K.mean(loss)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


#src: https://github.com/maciejkula/triplet_recommendations_keras
def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)


def create_triplet_loss_model_with_loss_fn(base_model, input_shape):
    anchor_input = Input(shape=input_shape)
    positive_input = Input(shape=input_shape)
    negative_input = Input(shape=input_shape)
    anchor_x = base_model(anchor_input)
    positive_x = base_model(positive_input)
    negative_x = base_model(negative_input)
    loss = Lambda(best_triplet_loss)([anchor_x, positive_x, negative_x])
    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=loss)
    model.compile(loss=identity_loss, optimizer=Adam(0.000003))
    return model, base_model


def build_embedding_network_base(sample_rate):
    input_shape = (1, sample_rate)
    inputs = Input(shape=input_shape)
    spectrogram = Melspectrogram(sr=sample_rate, n_dft=1024, input_shape=input_shape,
                                 return_decibel_melgram=True, trainable_kernel=False)(inputs)
    # spectrogram = AdditiveNoise(0.2)(spectrogram)
    spectrogram = Normalization2D(str_axis='freq')(spectrogram)
    spectrogram = Dropout(0.4)(spectrogram)
    convo = Convolution2D(96, 6, strides=2, activation='relu')(spectrogram)
    convo = Convolution2D(96, 1, strides=1, activation='relu')(convo)
    convo = BatchNormalization()(convo)
    convo = MaxPooling2D((2, 2), strides=2)(convo)
    convo = Dropout(0.4)(convo)
    convo = Convolution2D(256, 4, strides=2, activation='relu')(convo)
    convo = Convolution2D(256, 1, strides=1, activation='relu')(convo)
    convo = BatchNormalization()(convo)
    convo = MaxPooling2D((2, 2), strides=2)(convo)
    convo = Dropout(0.4)(convo)
    convo = Convolution2D(384, 2, strides=1, activation='relu')(convo)
    convo = Convolution2D(384, 1, strides=1, activation='relu')(convo)
    convo = BatchNormalization()(convo)
    # convo = MaxPooling2D((3, 3), strides=2)(convo)
    # convo = Convolution2D(256, (3, 3), strides=1)(convo)
    # convo = Convolution2D(256, (1, 1), strides=1)(convo)
    convo = AveragePooling2D((6, 3), strides=1)(convo)
    convo = Dropout(0.4)(convo)
    convo = BatchNormalization()(convo)
    # convo = Dense(1024, activation='sigmoid')(convo)
    # model = Model(inputs, convo)
    # model.compile(optimizer='sgd', loss='mse')
    return Model(inputs, convo)


def build_encoder_model_with_loss_fn():
    sample_rate = 44100
    input_shape = (1, sample_rate)
    embedding_network = build_embedding_network_base(sample_rate)
    triplet_loss_model, embedding_model = create_triplet_loss_model_with_loss_fn(embedding_network, input_shape)
    return triplet_loss_model, embedding_model


def batch_generator(hdf, epoch_count):
    while True:
        for idx in range(epoch_count):
            df_keys = hdf.keys()
            shuffled_keys = hdf.keys()
            shuffle(shuffled_keys)
            for (positive_df_key, negative_df_key) in zip(df_keys, shuffled_keys):
                # print('Reading and training df: ' + positive_df_key)
                # print('Anti-set: ' + negative_df_key)
                positive_df = hdf.get(positive_df_key)
                negative_df = hdf.get(negative_df_key)
                query_x = np.array(positive_df['feature'].tolist())
                positive_x = np.array(positive_df['feature'].copy(deep=True).tolist())
                shuffle(positive_x)
                negative_x = np.array(negative_df['feature'].tolist())
                min_dim = min(query_x.shape[0], negative_x.shape[0])
                trimmed_query_x = query_x[:min_dim, ]
                trimmed_positive_x = positive_x[:min_dim, ]
                trimmed_negative_x = negative_x[:min_dim, ]
                reshaped_query_x = trimmed_query_x[:, np.newaxis, :]
                reshaped_positive_x = trimmed_positive_x[:, np.newaxis, :]
                reshaped_negative_x = trimmed_negative_x[:, np.newaxis, :]
                # y = np.zeros((min_dim, 1))
                y = np.array([1, 0] * min_dim).reshape(min_dim, 2)
                yield [reshaped_query_x, reshaped_positive_x, reshaped_negative_x], y


triplet_embedding_model, embedding_model = build_encoder_model_with_loss_fn()
triplet_embedding_model.summary()
embedding_model.summary()

with pd.HDFStore('songs_training_data_44100_raw_parts.h5') as hdf:
    triplet_embedding_model.fit_generator(batch_generator(hdf, 10), steps_per_epoch=len(hdf.keys()), epochs=10)



embedding_model.save('model_raw_22050_nnfp_embedding_multi_04.h5')
triplet_embedding_model.save('model_raw_22050_nnfp_triplet_embedding_multi_04.h5')
# triplet_embedding_model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5, clipnorm=1.0), metrics=['accuracy'])
# model_metric.save('model_raw_22050_nnfp_metric_multi.h5')
# triplet_embedding_model.layers[3].save('model_raw_22050_nnfp_embedding.h5')


from keras.models import load_model
embedding_model = load_model('model_raw_22050_nnfp_embedding_multi.h5',
                       custom_objects={'Melspectrogram': Melspectrogram, 'tf': tf})
triplet_embedding_model = load_model('model_raw_22050_nnfp_triplet_embedding_multi.h5',
                       custom_objects={'Melspectrogram': Melspectrogram, 'tf': tf})

for l in embedding_model.layers:
    l.trainable = False


def read_encoding_df(hdf, embedding_model, encoder, df_key):
    df = hdf.get(df_key)
    X = np.array(df['feature'].tolist())
    reshaped_x = X[:, np.newaxis, :]
    output_x = embedding_model.predict(reshaped_x)
    reshaped_output_x = output_x.reshape(len(output_x), -1)
    # y = np_utils.to_categorical(encoder.transform(df['label'].tolist()), num_classes=encoder.classes_.shape[0])
    # return pd.DataFrame({'feature': np.array(reshaped_output_x).tolist(), 'label': np.array(y).tolist()})
    y = encoder.transform(df['label'].tolist())
    return pd.DataFrame({'feature': np.array(reshaped_output_x).tolist(), 'label': y})


def build_knn_training_data():
    with pd.HDFStore('songs_training_data_22050_raw_parts.h5') as hdf:
        read_encoding_df_fn = partial(read_encoding_df, hdf, embedding_model, encoder)
        actual_training_datas = [read_encoding_df_fn(df_key) for df_key in hdf.keys()]
        df = pd.concat(actual_training_datas)
        df.columns = ['feature', 'label']
        return df


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1)
df = build_knn_training_data()
knn.fit(list(df['feature']), list(df['label']))
knn.score(list(df['feature']), list(df['label']))


import joblib
joblib.dump(knn, 'knn_multi_02.joblib')
# knn = joblib.load('knn.joblib')