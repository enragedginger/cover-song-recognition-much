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
import h5py

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
    # model.compile(loss=identity_loss, optimizer=Adam(0.000003))
    model.compile(loss=identity_loss, optimizer=Adam())
    return model, base_model


def build_mfcc_embedding_network_base(mfcc_cnt):
    input_shape = (1, mfcc_cnt, 44)
    inputs = Input(shape=input_shape)
    convo = Convolution2D(8, (3, 3), strides=1, activation='relu', data_format='channels_first')(inputs)
    convo = BatchNormalization()(convo)
    convo = MaxPooling2D((2, 2), strides=1, data_format='channels_first')(convo)
    convo = Convolution2D(16, (3, 3), strides=2, activation='relu', data_format='channels_first')(convo)
    convo = BatchNormalization()(convo)
    convo = MaxPooling2D((2, 2), strides=2, data_format='channels_first')(convo)
    convo = Convolution2D(32, (3, 3), strides=2, activation='relu', data_format='channels_first')(convo)
    convo = BatchNormalization()(convo)
    convo = MaxPooling2D((2, 2), strides=2, data_format='channels_first')(convo)
    # convo = Convolution2D(32, (7, 7), strides=2, activation='relu', data_format='channels_first')(convo)
    # convo = BatchNormalization()(convo)
    # convo = AveragePooling2D((2, 2), strides=1, data_format='channels_first')(convo)
    # idx = 0
    # while idx < 8:
    #     idx += 1
    #     convo = Convolution2D(1, (3, 1), strides=1, activation='relu', data_format='channels_first')(convo)
    #     convo = Convolution2D(1, (1, 3), strides=1, activation='relu', data_format='channels_first')(convo)
    #     # convo = BatchNormalization()(convo)
    #     convo = AveragePooling2D((3, 3), strides=1, data_format='channels_first')(convo)
    convo = Flatten()(convo)
    convo = Dense(250, activation='softmax')(convo)
    # model = Model(inputs, convo)
    # model.compile(optimizer='sgd', loss='mse')
    return Model(inputs, convo)


def build_encoder_model_with_loss_fn(mfcc_cnt):
    input_shape = (1, mfcc_cnt, 44)
    embedding_network = build_mfcc_embedding_network_base(mfcc_cnt)
    triplet_loss_model, embedding_model = create_triplet_loss_model_with_loss_fn(embedding_network, input_shape)
    return triplet_loss_model, embedding_model


def pick_rando(items, avoidance_index):
    indices = range(len(items))
    idx = randy.choice(indices)
    if len(items) > 1:
        while avoidance_index == idx:
            idx = randy.choice(indices)
    return items[idx]


def batch_generator(df_keys, training_data_filename):
    idx = 0
    while True:
        positive_df_key = df_keys[idx]
        negative_df_key = pick_rando(df_keys, idx)
        # print('Reading and training df: ' + positive_df_key)
        # print('Anti-set: ' + negative_df_key)
        with h5py.File(training_data_filename, 'r') as hdf:
            query_x = hdf[positive_df_key]['feature'][()]
            positive_x = query_x.copy()
            negative_x = hdf[negative_df_key]['feature'][()]
        randy.shuffle(positive_x)
        min_dim = min(query_x.shape[0], negative_x.shape[0])
        trimmed_query_x = query_x[:min_dim, ]
        trimmed_positive_x = positive_x[:min_dim, ]
        trimmed_negative_x = negative_x[:min_dim, ]
        reshaped_query_x = trimmed_query_x[:, np.newaxis, :]
        reshaped_positive_x = trimmed_positive_x[:, np.newaxis, :]
        reshaped_negative_x = trimmed_negative_x[:, np.newaxis, :]
        # y = np.zeros((min_dim, 1))
        y = np.array([1, 0] * min_dim).reshape(min_dim, 2)
        idx += 1
        if idx >= len(df_keys):
            idx = 0
        yield [reshaped_query_x, reshaped_positive_x, reshaped_negative_x], y


# Build the models and print the summary as a sanity check
triplet_embedding_model, embedding_model = build_encoder_model_with_loss_fn(40)
triplet_embedding_model.summary()
embedding_model.summary()


training_data_filename = 'songs_training_data_22050_mfcc_2d_parts.h5'
validation_data_filename = 'songs_validation_data_22050_mfcc_2d_parts.h5'


# max_queue_size=1, workers=0, steps_per_epoch=1,
#with pd.HDFStore(training_data_filename) as hdf:
with h5py.File(training_data_filename, 'a') as hdf:
    hdf_keys = list(hdf.keys())
    triplet_embedding_model.fit_generator(batch_generator(hdf_keys, training_data_filename),
                                          validation_data=batch_generator(hdf_keys, validation_data_filename),
                                          validation_steps=len(hdf_keys),
                                          steps_per_epoch=len(hdf_keys), epochs=100, use_multiprocessing=True)



# Example code for saving the models to disk
embedding_model.save('model_raw_22050_nnfp_embedding_mmfc_01.h5')
triplet_embedding_model.save('model_raw_22050_nnfp_triplet_embedding_mmfc_01.h5')


# Example code for reading the models from disk with custom loss functions
from keras.models import load_model
embedding_model = load_model('model_raw_22050_nnfp_embedding_mmfc_01.h5',
                       custom_objects={'tf': tf, 'identity_loss': identity_loss})
triplet_embedding_model = load_model('model_raw_22050_nnfp_triplet_embedding_mmfc_01.h5',
                       custom_objects={'tf': tf, 'identity_loss': identity_loss})


# Unset the trainable parameters after reading the model in from disk
for l in embedding_model.layers:
    l.trainable = False


# Read data for usage by KNN
def read_encoding_df(hdf, embedding_model, encoder, df_key):
    X = hdf[df_key]['feature'][()]
    label = hdf[df_key]['label'][()]
    reshaped_x = X[:, np.newaxis, :]
    output_x = embedding_model.predict(reshaped_x)
    reshaped_output_x = output_x.reshape(len(output_x), -1)
    # y = np_utils.to_categorical(encoder.transform(df['label'].tolist()), num_classes=encoder.classes_.shape[0])
    # return pd.DataFrame({'feature': np.array(reshaped_output_x).tolist(), 'label': np.array(y).tolist()})
    y = encoder.transform([label] * len(X))
    return pd.DataFrame({'feature': np.array(reshaped_output_x).tolist(), 'label': y})


# Run all of the songs through our model, generate embeddings, and export as a giant DF
def build_knn_data(data_filename):
    # with pd.HDFStore('songs_training_data_22050_mfcc_parts.h5') as hdf:
    with h5py.File(data_filename, 'a') as hdf:
        read_encoding_df_fn = partial(read_encoding_df, hdf, embedding_model, encoder)
        actual_training_datas = [read_encoding_df_fn(df_key) for df_key in hdf.keys()]
        df = pd.concat(actual_training_datas)
        df.columns = ['feature', 'label']
        return df


# train and test the knn. This could take a long time.
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1)
train_df = build_knn_data(training_data_filename)
knn.fit(list(train_df['feature']), train_df['label'])
test_df = build_knn_data(validation_data_filename)
knn.score(list(test_df['feature']), list(test_df['label']))


import joblib
joblib.dump(knn, 'knn_mfcc_01.joblib')
knn = joblib.load('knn_mfcc_01.joblib')