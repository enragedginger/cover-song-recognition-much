import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.models import load_model
from audio_much.core import feature_gen_row
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from kapre.time_frequency import Melspectrogram
from kapre.augmentation import AdditiveNoise
from kapre.utils import Normalization2D
import collections

encoder = LabelEncoder()
encoder.classes_ = np.load('songs_training_data_classes.npy')

basic_model = load_model('basic_af_model.h5')
raw_model = load_model('melspectorgram_raw_22050_model.h5',
                       custom_objects={'Melspectrogram': Melspectrogram,
                                       'AdditiveNoise': AdditiveNoise,
                                       'Normalization2D': Normalization2D})
raw_embedded_knn_model = load_model('model_raw_22050_nnfp_knn.h5', custom_objects={'Melspectrogram': Melspectrogram})


def do_prediction(song_path, target_sample_rate=22050, n=1):
    model = basic_model
    X, sample_rate = librosa.load(song_path, sr=target_sample_rate, res_type='kaiser_fast')
    feature = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    y_prob = model.predict(feature.reshape((1, 40)))
    top_n_indices = np.argpartition(y_prob[0], -n)[-n:]
    top_n_probs = y_prob[0][top_n_indices]
    top_n_classes = encoder.classes_[top_n_indices]
    return pd.DataFrame({'class': top_n_classes, 'prob': top_n_probs})


def grab_top_n_probs(n, y_prob_row):
    top_n_indices = np.argpartition(y_prob_row, -n)[-n:]
    top_n_probs = y_prob_row[top_n_indices]
    top_n_classes = encoder.classes_[top_n_indices]
    return pd.DataFrame({'class': top_n_classes.tolist(), 'prob': top_n_probs.tolist()})


def do_prediction_raw(song_path, model=raw_embedded_knn_model, target_sample_rate=22050, n=1):
    X, sample_rate = librosa.load(song_path, sr=target_sample_rate, res_type='kaiser_fast')
    len_second = 1.0
    split = int(target_sample_rate * len_second)
    split_remainder = int(X.shape[0] % split)
    split_count = int(X[:-split_remainder].shape[0] / split)
    audio_segments = np.split(X[:-split_remainder], split_count)
    float_audio_segments = [x.astype('float') for x in audio_segments]
    series_audio_segments = np.array(float_audio_segments)
    features = series_audio_segments[:, np.newaxis, :]
    y_prob = model.predict(features)
    result_dfs = [grab_top_n_probs(n, y_prob[idx]) for idx in np.arange(y_prob.shape[0])]
    result_df = pd.concat(result_dfs)
    grouped_results_df = result_df.groupby('class', as_index=False).agg({'prob': 'sum'})
    return grouped_results_df.sort_values(['prob'], ascending=False)


def do_prediction_knn(embedding_model, knn, song_path, target_sample_rate=22050):
    X, sample_rate = librosa.load(song_path, sr=target_sample_rate, res_type='kaiser_fast')
    len_second = 1.0
    split = int(target_sample_rate * len_second)
    split_remainder = int(X.shape[0] % split)
    split_count = int(X[:-split_remainder].shape[0] / split)
    audio_segments = np.split(X[:-split_remainder], split_count)
    float_audio_segments = [x.astype('float') for x in audio_segments]
    series_audio_segments = np.array(float_audio_segments)
    raw_features = series_audio_segments[:, np.newaxis, :]
    features = embedding_model.predict(raw_features)
    reshaped_features = features.reshape(len(features), -1)
    y_prob = knn.predict(reshaped_features)
    # classes = [encoder.classes_[y_prob[idx].argmax()] for idx in np.arange(y_prob.shape[0])]
    classes = [encoder.classes_[y_prob[idx]] for idx in np.arange(y_prob.shape[0])]
    return collections.Counter(classes)


do_prediction_knn(embedding_model, knn, '/Users/hopper/Desktop/music/Pickin\' On/Pickin\' On Led Zeppelin/03_-_ramble_on.mp3')
do_prediction_knn(embedding_model, knn, '/Users/hopper/Music/GarageBand/SmellsLikeTeenSpirit.mp3')
do_prediction_knn(embedding_model, knn, '/Users/hopper/Music/GarageBand/ComeAsYouAre.mp3')
do_prediction_knn(embedding_model, knn, '/Users/hopper/Music/GarageBand/SevenNationArmy.mp3')
do_prediction_knn(embedding_model, knn, '/Users/hopper/Desktop/music/Rage Against the Machine/Renegades/06 I\'m Housin\'.wma')

do_prediction('/Users/hopper/Desktop/music/Nirvana/Nevermind/01 Smells Like Teen Spirit.wma')


song_path = '/Users/hopper/Music/GarageBand/SmellsLikeTeenSpirit.mp3'
target_sample_rate = 22050
n = 10
model = raw_model
X, sample_rate = librosa.load(song_path, sr=target_sample_rate, res_type='kaiser_fast')
len_second = 1.0
split = int(target_sample_rate * len_second)
split_remainder = int(X.shape[0] % split)
split_count = int(X[:-split_remainder].shape[0] / split)
audio_segments = np.split(X[:-split_remainder], split_count)
float_audio_segments = [x.astype('float') for x in audio_segments]
series_audio_segments = np.array(float_audio_segments)
features = series_audio_segments[:, np.newaxis, :]
y_prob = model.predict(features)

for idx in y_prob.shape[0]:
    top_n_indices = np.argpartition(y_prob[idx], -n)[-n:]
    top_n_probs = y_prob[idx][top_n_indices]
    top_n_classes = encoder.classes_[top_n_indices]
    pd.DataFrame({'class': top_n_classes.tolist(), 'prob': top_n_probs.tolist()})



X, sample_rate = librosa.load('/Users/hopper/Music/GarageBand/SmellsLikeTeenSpirit.mp3', sr=22050, res_type='kaiser_fast', mono=True)
len_second = 1.0
X = X[:int(22050 * len_second)]
X = X[np.newaxis, :]
feature = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
y_prob = model.predict(feature.reshape((1, 40)))
top_n_indices = np.argpartition(y_prob[0], -n)[-n:]
top_n_probs = y_prob[0][top_n_indices]
top_n_classes = encoder.classes_[top_n_indices]


original_audio, sample_rate = librosa.load('/Users/hopper/Music/GarageBand/SmellsLikeTeenSpirit.mp3', res_type='kaiser_fast')