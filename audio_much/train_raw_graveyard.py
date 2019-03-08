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


def custom_triplet_loss(x):
    anchor, positive, negative = x
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
    basic_loss = tf.subtract(neg_dist, pos_dist)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    return loss


# Source: https://github.com/davidsandberg/facenet/blob/master/src/facenet.py
def triplet_loss(x):
    ALPHA = 0.2 # triplet loss parameter
    anchor, positive, negative = x
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), ALPHA)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    return loss


#src: https://github.com/maciejkula/triplet_recommendations_keras
def bpr_triplet_loss(X):
    user_latent, positive_item_latent, negative_item_latent = X
    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))
    return loss


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


def create_triplet_loss_model(base_model, input_shape):
    anchor_input = Input(shape=input_shape)
    positive_input = Input(shape=input_shape)
    negative_input = Input(shape=input_shape)
    reduce_sum = Dense(1, activation='linear', kernel_initializer='ones', bias_initializer='zeros', name='reduce_sum')
    anchor_x = base_model(anchor_input)
    positive_x = base_model(positive_input)
    negative_x = base_model(negative_input)
    anchor_x = Reshape((1, 1, 102))(anchor_x)
    positive_x = Reshape((1, 1, 102))(positive_x)
    negative_x = Reshape((1, 1, 102))(negative_x)
    positive_distance = subtract([anchor_x, positive_x])
    negative_distance = subtract([anchor_x, negative_x])
    positive_distance = Lambda(lambda val: (val) ** 2)(positive_distance)
    negative_distance = Lambda(lambda val: (val) ** 2)(negative_distance)
    positive_distance = reduce_sum(positive_distance)
    negative_distance = reduce_sum(negative_distance)
    positive_distance = Lambda(lambda val: K.sqrt(val + K.epsilon()))(positive_distance)
    negative_distance = Lambda(lambda val: K.sqrt(val + K.epsilon()))(negative_distance)
    # d = Concatenate(axis=1)([positive_distance, negative_distance])
    d = concatenate([positive_distance, negative_distance])
    d = Activation('softmax')(d)
    d = Flatten()(d)
    # d = Reshape((1, 1))(d)
    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=d)
    metric = Model(inputs=anchor_input, outputs=anchor_x)
    for l in model.layers:
        if l.name == 'reduce_sum':
            print('reduce sum')
            l.trainable = False
    return model, metric


def build_embedding_network_base(sample_rate):
    input_shape = (1, sample_rate)
    inputs = Input(shape=input_shape)
    spectrogram = Melspectrogram(sr=sample_rate, n_dft=1024, input_shape=input_shape,
                                 return_decibel_melgram=True, trainable_kernel=False)(inputs)
    spectrogram = AdditiveNoise(0.2)(spectrogram)
    spectrogram = Normalization2D(str_axis='freq')(spectrogram)
    convo = Convolution2D(1, 3, strides=1, activation='relu')(spectrogram)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(3, 1, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(1, 3, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(3, 1, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(1, 3, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(3, 1, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(1, 3, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(3, 1, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(1, 3, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(3, 1, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(1, 3, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(3, 1, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(1, 3, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(3, 1, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(1, 3, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(3, 1, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(1, 3, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(3, 1, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(1, 3, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(3, 1, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(1, 3, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(3, 1, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(1, 3, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    convo = Convolution2D(3, 1, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    # convo = MaxPooling2D((2, 2), strides=1)(convo)
    convo = Convolution2D(1, 3, strides=1, activation='relu')(convo)
    convo = Lambda(lambda x: K.l2_normalize(x, axis=1))(convo)
    # convo = Convolution2D(3, 1, strides=1, activation='relu')(convo)
    convo = AveragePooling2D((1, 18), strides=1)(convo)
    # convo = BatchNormalization()(convo)
    # convo = Dense(1024, activation='sigmoid')(convo)
    # model = Model(inputs, convo)
    # model.compile(optimizer='sgd', loss='mse')
    return Model(inputs, convo)


# Builds an embedding for each example (i.e., positive, negative, anchor)
# Then calculates the triplet loss between their embedding.
# Then applies identity loss on the triplet loss value to minimize it on training.
def build_encoder_model():
    sample_rate = 22050
    input_shape = (1, sample_rate)
    # Create Common network to share the weights along different examples (+/-/Anchor)
    embedding_network = build_embedding_network_base(sample_rate)
    triplet_loss_model, model_metric = create_triplet_loss_model(embedding_network, input_shape)
    # distance = Lambda(custom_triplet_loss, output_shape=(1,))([anchor_embedding, positive_embedding, negative_embedding])
    # model = Model(inputs=Input(shape=input_shape), outputs=embedding_network)
    triplet_loss_model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5, clipnorm=1.0), metrics=['accuracy'])
    # triplet_loss_model.fit([train_1, train_2, train_3], train_y, batch_size=16, epochs=5)
    return triplet_loss_model, embedding_network, model_metric


def build_knn(model, output_size):
    # Flatten feature vector
    flat_dim_size = np.prod(model.output_shape[1:])
    x = Reshape(target_shape=(flat_dim_size,),
                name='features_flat')(model.get_output_at(0))
    # Dot product between feature vector and reference vectors
    x = Dense(units=output_size,
              name='dense_1')(x)
    classifier = Model(inputs=[model.get_input_at(0)], outputs=x)
    classifier.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier


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


def knn_batch_generator(hdf, epoch_count):
    while True:
        for idx in range(epoch_count):
            for df_key in hdf.keys():
                df = hdf.get(df_key)
                X = np.array(df['feature'].tolist())
                reshaped_x = X[:, np.newaxis, :]
                y = np_utils.to_categorical(encoder.transform(df['label'].tolist()), num_classes=encoder.classes_.shape[0])
                yield reshaped_x, y


def save_training_data(sample_rate, out_file):
    training_data = pd.read_csv('songs_training.csv', header=0)
    feature_gen_row_fn = partial(read_encoding_df, sample_rate)
    actual_training_data = training_data.apply(feature_gen_row_fn, axis=1)
    actual_training_data.columns = ['feature', 'label']
    df = pd.DataFrame(actual_training_data.values.tolist())
    df.columns = ['feature', 'label']
    df = df.dropna()
    df.to_pickle(out_file)

