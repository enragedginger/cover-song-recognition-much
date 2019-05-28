import click
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.models import load_model
from audio_much.core import build_float_audio_mfcc_2d_segments, build_audio_feature_sequences, build_beat_audio_feature_sequences
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import collections
import h5py

encoder = LabelEncoder()
encoder.classes_ = np.load('songs_training_data_classes.npy')

lstm_model = load_model('model_raw_22050_lstm_02.h5')


# def do_prediction(song_path, target_sample_rate=22050, n=1):
#     model = basic_model
#     X, sample_rate = librosa.load(song_path, sr=target_sample_rate, res_type='kaiser_fast')
#     feature = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
#     y_prob = model.predict(feature.reshape((1, 40)))
#     top_n_indices = np.argpartition(y_prob[0], -n)[-n:]
#     top_n_probs = y_prob[0][top_n_indices]
#     top_n_classes = encoder.classes_[top_n_indices]
#     return pd.DataFrame({'class': top_n_classes, 'prob': top_n_probs})
#
#
# def grab_top_n_probs(n, y_prob_row):
#     top_n_indices = np.argpartition(y_prob_row, -n)[-n:]
#     top_n_probs = y_prob_row[top_n_indices]
#     top_n_classes = encoder.classes_[top_n_indices]
#     return pd.DataFrame({'class': top_n_classes.tolist(), 'prob': top_n_probs.tolist()})
#
#
# def do_prediction_raw(song_path, model=raw_embedded_knn_model, target_sample_rate=22050, n=1):
#     X, sample_rate = librosa.load(song_path, sr=target_sample_rate, res_type='kaiser_fast')
#     len_second = 1.0
#     split = int(target_sample_rate * len_second)
#     split_remainder = int(X.shape[0] % split)
#     split_count = int(X[:-split_remainder].shape[0] / split)
#     audio_segments = np.split(X[:-split_remainder], split_count)
#     float_audio_segments = [x.astype('float') for x in audio_segments]
#     series_audio_segments = np.array(float_audio_segments)
#     features = series_audio_segments[:, np.newaxis, :]
#     y_prob = model.predict(features)
#     result_dfs = [grab_top_n_probs(n, y_prob[idx]) for idx in np.arange(y_prob.shape[0])]
#     result_df = pd.concat(result_dfs)
#     grouped_results_df = result_df.groupby('class', as_index=False).agg({'prob': 'sum'})
#     return grouped_results_df.sort_values(['prob'], ascending=False)
#
#
# def do_prediction_knn(embedding_model, knn, song_path, target_sample_rate=22050):
#     X, sample_rate = librosa.load(song_path, sr=target_sample_rate, res_type='kaiser_fast')
#     len_second = 1.0
#     split = int(target_sample_rate * len_second)
#     split_remainder = int(X.shape[0] % split)
#     split_count = int(X[:-split_remainder].shape[0] / split)
#     audio_segments = np.split(X[:-split_remainder], split_count)
#     float_audio_segments = [x.astype('float') for x in audio_segments]
#     series_audio_segments = np.array(float_audio_segments)
#     raw_features = series_audio_segments[:, np.newaxis, :]
#     features = embedding_model.predict(raw_features)
#     reshaped_features = features.reshape(len(features), -1)
#     y_prob = knn.predict(reshaped_features)
#     # classes = [encoder.classes_[y_prob[idx].argmax()] for idx in np.arange(y_prob.shape[0])]
#     classes = [encoder.classes_[y_prob[idx]] for idx in np.arange(y_prob.shape[0])]
#     return collections.Counter(classes)
#
#
# def do_prediction_knn_mfcc(embedding_model, knn, song_path, target_sample_rate=22050):
#     X, sample_rate = librosa.load(song_path, sr=target_sample_rate, res_type='kaiser_fast')
#     float_audio_segments = build_float_audio_mfcc_2d_segments(X, target_sample_rate)
#     series_audio_segments = np.array(float_audio_segments)
#     raw_features = series_audio_segments[:, np.newaxis, :]
#     features = embedding_model.predict(raw_features)
#     reshaped_features = features.reshape(len(features), -1)
#     y_prob = knn.predict(reshaped_features)
#     # classes = [encoder.classes_[y_prob[idx].argmax()] for idx in np.arange(y_prob.shape[0])]
#     classes = [encoder.classes_[y_prob[idx]] for idx in np.arange(y_prob.shape[0])]
#     counts = collections.Counter(classes)
#     [print(str(k) + ': ' + str(v)) for k, v in counts.items()]


def do_prediction_lstm(lstm_model, song_path, target_sample_rate=22050):
    X, sample_rate = librosa.load(song_path, sr=target_sample_rate, res_type='kaiser_fast')
    hop_length = 512
    tempo, beats = librosa.beat.beat_track(y=X, sr=target_sample_rate, hop_length=hop_length)
    float_audio_segments = build_beat_audio_feature_sequences(X, target_sample_rate, tempo)
    timeseries_length = 16
    batch_size = float_audio_segments.shape[0] - timeseries_length
    features = np.zeros((batch_size, timeseries_length, 12), dtype=np.float64)
    for batch_idx in range(0, batch_size):
        for timeseries_idx in range(0, timeseries_length):
            features[batch_idx, timeseries_idx, :] = float_audio_segments[batch_idx + timeseries_idx]
    res = lstm_model.predict(features)
    classes = [encoder.classes_[np.argmax(res[idx])] for idx in np.arange(res.shape[0])]
    counts = collections.Counter(classes)
    print("Tempo: {:.2f}".format(tempo))
    [(max_label, occurrences)] = counts.most_common(1)
    print("Top song: {:s} - {:d} / {:d} = {:.2f}".format(max_label, occurrences, res.shape[0], occurrences / res.shape[0]))
    [print(str(k) + ': ' + str(v)) for k, v in counts.items()]


# librosa.load('/home/stephen/Desktop/GarageBand/ComeAsYouAre.mp3', sr=22050, res_type='kaiser_fast')
#
# do_prediction_lstm(lstm_model, '/Users/hopper/Desktop/validation_music/Pickin\' On Led Zeppelin/07. Led Zeppelin - Ramble On.mp3')
# do_prediction_lstm(lstm_model, '/home/stephen/Desktop/GarageBand/SmellsLikeTeenSpirit.mp3')
# do_prediction_lstm(lstm_model, '/home/stephen/Desktop/GarageBand/ComeAsYouAre.mp3')
# do_prediction_lstm(lstm_model, '/home/stephen/Desktop/GarageBand/SevenNationArmy.mp3')
# do_prediction_lstm(lstm_model, '/home/stephen/Desktop/music/training/Rage Against the Machine/Renegades/06 I\'m Housin\'.wma')
#
# do_prediction_knn_mfcc(embedding_model, knn, "/home/stephen/Desktop/music/validation/Pickin' On/Pickin' On Led Zeppelin/02_-_kashmir.mp3")
#
#
# do_prediction_lstm(lstm_model, '/Users/hopper/Music/GarageBand/SmellsLikeTeenSpirit.mp3')
# do_prediction_lstm(lstm_model, '/Users/hopper/Music/GarageBand/ComeAsYouAre.mp3')
# do_prediction_lstm(lstm_model, '/Users/hopper/Music/GarageBand/SevenNationArmy.mp3')
# do_prediction_lstm(lstm_model, '/Users/hopper/Desktop/music_subset/wonderwall.mp3')
# do_prediction_lstm(lstm_model, '/Users/hopper/Desktop/music/Rage Against the Machine/Renegades/06 I\'m Housin\'.wma')


@click.command()
@click.argument('song_path')
def main(song_path):
    do_prediction_lstm(lstm_model, song_path, 44100)


if __name__ == "__main__":
    main()
a