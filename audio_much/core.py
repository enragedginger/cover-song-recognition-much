import os
import pandas as pd
import glob
import h5py
import csv

import librosa
# import librosa.display
import numpy as np
import re
from functools import partial
from sklearn.preprocessing import LabelEncoder

root_music_dir = '/Users/hopper/Desktop/music_subset'
root_music_validation_dir = '/Users/hopper/Desktop/validation_music'
# root_music_dir='/home/stephen/Desktop/music/training'
training_csv = 'songs_training.csv'
validation_csv = 'songs_validation.csv'


tempo_map = {
    "testify": 118,
    "bound_for_the_floor": 119,
    "dont_fear_the_reaper": 141,
    "in_bloom": 78,
    "lake_of_fire": 145,
    "aqualung": 123,
    "the_trooper": 160,
    "you_gotta_fight_for_your_right_to_party": 133,
    "killing_in_the_name": 89,
    "rock_you_like_a_hurricane": 126,
    "highway_to_hell": 116,
    "desire": 109,
    "woman": 113,
    "paranoid": 163,
    "wonderwall": 175,
    "smells_like_teen_spirit": 117,
    "communication_breakdown": 175,
    "californication": 96,
    "come_as_you_are": 120,
    "seven_nation_army": 124,
    "run_to_the_hills": 174,
    "hit_me_with_your_best_shot": 127
}


def write_songs_csv(filename, music_dir):
    with open(filename, 'w') as out_file:
        writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(['name', 'location', 'tempo'])
        for root, directories, filenames in os.walk(music_dir):
            for filename in filenames:
                if re.match('^.*\.(wma|WMA|mp3|MP3|wav|WAV|m4a|M4A)$', filename):
                    song_name = ('.').join(filename.split('.')[:-1])
                    tempo = tempo_map[song_name]
                    writer.writerow([song_name, os.path.join(root, filename), tempo])


def write_training_csv():
    write_songs_csv(training_csv, root_music_dir)


def write_validation_csv():
    write_songs_csv(validation_csv, root_music_validation_dir)


def feature_gen_row(target_sample_rate, row):
    try:
        print('generating features for: ' + row['location'])
        original_audio, sample_rate = librosa.load(row['location'], res_type='kaiser_fast')
        if target_sample_rate != sample_rate:
            print('rescaling from ' + str(sample_rate) + ' to ' + str(target_sample_rate))
            X = librosa.core.resample(original_audio, sample_rate, target_sample_rate)
        else:
            X = original_audio
        feature = np.mean(librosa.feature.mfcc(y=X, sr=target_sample_rate, n_mfcc=40).T, axis=0)
        return [feature, row['name']]
    except Exception as e:
        print('error while generating features for: ' + row['location'])
        print(e)
        return [None, row['name']]


def build_float_audio_segments(X, target_sample_rate):
    len_second = 1.0
    split = int(target_sample_rate * len_second)
    split_remainder = int(X.shape[0] % split)
    split_count = int(X[:-split_remainder].shape[0] / split) if split_remainder != 0 else int(X[:].shape[0] / split)
    audio_segments = np.split(X[:-split_remainder], split_count) if split_remainder != 0 else np.split(X[:], split_count)
    return [x.astype('float') for x in audio_segments]


def build_float_audio_mfcc_segments(X, target_sample_rate):
    len_second = 1.0
    split = int(target_sample_rate * len_second)
    split_remainder = int(X.shape[0] % split)
    split_count = int(X[:-split_remainder].shape[0] / split) if split_remainder != 0 else int(X[:].shape[0] / split)
    audio_segments = np.split(X[:-split_remainder], split_count) if split_remainder != 0 else np.split(X[:], split_count)
    return np.array([np.mean(librosa.feature.mfcc(y=x.astype('float'), sr=target_sample_rate, n_mfcc=40).T, axis=0) for x in audio_segments])


def build_float_audio_mfcc_2d_segments(X, target_sample_rate):
    len_second = 1.0
    split = int(target_sample_rate * len_second)
    split_remainder = int(X.shape[0] % split)
    split_count = int(X[:-split_remainder].shape[0] / split) if split_remainder != 0 else int(X[:].shape[0] / split)
    audio_segments = np.split(X[:-split_remainder], split_count) if split_remainder != 0 else np.split(X[:], split_count)
    return np.array([librosa.feature.mfcc(y=x.astype('float'), sr=target_sample_rate, n_mfcc=40) for x in audio_segments])


def build_audio_feature_sequences(X, target_sample_rate):
    hop_length = 4096
    mfcc = librosa.feature.mfcc(y=X, sr=target_sample_rate, hop_length=hop_length, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=X, sr=target_sample_rate, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=X, sr=target_sample_rate, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(y=X, sr=target_sample_rate, hop_length=hop_length)
    timeseries_length = mfcc.shape[1]
    tempo, beats = librosa.beat.beat_track(y=X, sr=target_sample_rate, hop_length=hop_length)
    features = np.zeros((timeseries_length, 34), dtype=np.float64)
    features[:, 0:13] = mfcc.T[0:timeseries_length, :]
    features[:, 13:14] = spectral_center.T[0:timeseries_length, :]
    features[:, 14:26] = chroma.T[0:timeseries_length, :]
    features[:, 26:33] = spectral_contrast.T[0:timeseries_length, :]
    features[:, 33] = [tempo] * timeseries_length
    return features


def build_beat_audio_feature_sequences(y, target_sample_rate, estimated_tempo):
    hop_length = 512
    # y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=target_sample_rate, start_bpm=estimated_tempo)
    # mfcc = librosa.feature.mfcc(y=y, sr=target_sample_rate, hop_length=hop_length, n_mfcc=13)
    # mfcc_delta = librosa.feature.delta(mfcc)
    # mfcc_feature_stack = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)
    chromagram = librosa.feature.chroma_cens(y=y, sr=target_sample_rate, hop_length=hop_length)
    beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.mean)
    # spectral_contrast = librosa.feature.spectral_contrast(y=y_harmonic, sr=target_sample_rate, hop_length=hop_length)
    # beat_spectral_contrast = librosa.util.sync(spectral_contrast, beat_frames, aggregate=np.mean)
    timeseries_length = beat_chroma.shape[1]
    features = np.zeros((timeseries_length, 12), dtype=np.float64)
    # features[:, 0:26] = mfcc_feature_stack.T[0:timeseries_length, :]
    # features[:, 26:38] = beat_chroma.T[0:timeseries_length, :]
    # features[:, 38:45] = beat_spectral_contrast.T[0:timeseries_length, :]
    features[:, 0:12] = beat_chroma.T[0:timeseries_length, :]
    return features


# https://github.com/keunwoochoi/kapre/blob/master/examples/prepare%20audio.ipynb
def feature_gen_row_raw(target_sample_rate, row):
    try:
        print('generating features for: ' + row['location'])
        X, sample_rate = librosa.load(row['location'], sr=target_sample_rate, res_type='kaiser_fast')
        len_second = 1.0
        split = int(target_sample_rate * len_second)
        split_remainder = int(X.shape[0] % split)
        split_count = int(X[:-split_remainder].shape[0] / split)
        audio_segments = np.split(X[:-split_remainder], split_count)
        float_audio_segments = [x.astype('float') for x in audio_segments]
        df = pd.DataFrame({
            # 'feature': pd.Series(float_audio_segments),
            'feature': [pd.Series(d) for d in float_audio_segments],
            'label': pd.Series(([row['name']] * len(audio_segments)), dtype='str')
        })
        return df
    except Exception as e:
        print('error while generating features for: ' + row['location'])
        print(e)
        return pd.DataFrame({'feature': [], 'label': pd.Series(dtype='str')})


def feature_gen_row_raw_save(target_sample_rate, hdf, idx, row):
    try:
        print('generating features for: ' + row['location'])
        X, sample_rate = librosa.load(row['location'], sr=target_sample_rate, res_type='kaiser_fast')
        # X_50, sample_rate_50 = librosa.load(row['location'], sr=target_sample_rate, res_type='kaiser_fast', offset=0.5)
        # X_25, sample_rate_25 = librosa.load(row['location'], sr=target_sample_rate, res_type='kaiser_fast', offset=0.25)
        for shift_idx in range(0, 1):
            print('shifting by ' + str(shift_idx) + ' steps')
            if shift_idx != 0:
                y = librosa.effects.pitch_shift(X, target_sample_rate, n_steps=shift_idx)
            else:
                y = X
            raw_audio = build_beat_audio_feature_sequences(y, target_sample_rate, row['tempo'])
            # raw_audio_full = np.append(np.append(raw_audio, raw_audio_50, axis=0), raw_audio_25, axis=0)
            raw_audio_full = raw_audio
            group_key = 'df_' + str(idx) + '_' + str(shift_idx)
            entry_group = hdf.create_group(group_key)
            entry_group.create_dataset('label', data=row['name'])
            entry_group.create_dataset('feature', data=raw_audio_full)
    except Exception as e:
        print('error while generating features for: ' + row['location'])
        print(e)
        raise e


def feature_gen_row_raw_validation_save(target_sample_rate, hdf, idx, row):
    try:
        print('generating features for: ' + row['location'])
        X, sample_rate = librosa.load(row['location'], sr=target_sample_rate, res_type='kaiser_fast')
        raw_audio = build_beat_audio_feature_sequences(X, target_sample_rate, row['tempo'])
        group_key = 'df_' + str(idx)
        entry_group = hdf.create_group(group_key)
        entry_group.create_dataset('label', data=row['name'])
        entry_group.create_dataset('feature', data=raw_audio)
    except Exception as e:
        print('error while generating features for: ' + row['location'])
        print(e)
        raise e


def save_training_data(sample_rate, out_file):
    training_data = pd.read_csv(training_csv, header=0)
    feature_gen_row_fn = partial(feature_gen_row, sample_rate)
    actual_training_data = training_data.apply(feature_gen_row_fn, axis=1)
    actual_training_data.columns = ['feature', 'label']
    df = pd.DataFrame(actual_training_data.values.tolist())
    df.columns = ['feature', 'label']
    df = df.dropna()
    df.to_pickle(out_file)


def save_training_data_raw(sample_rate, out_file):
    training_data = pd.read_csv(training_csv, header=0)
    # actual_training_data = training_data.apply(feature_gen_row_fn, axis=1)
    # df = pd.concat(actual_training_data.values.tolist())
    with h5py.File(out_file, 'a') as hdf:
        for idx in np.arange(training_data.shape[0]):
            feature_gen_row_raw_save(sample_rate, hdf, idx, training_data.iloc[idx])


def save_validation_data_raw(sample_rate, out_file):
    training_data = pd.read_csv(validation_csv, header=0)
    # actual_training_data = training_data.apply(feature_gen_row_fn, axis=1)
    # df = pd.concat(actual_training_data.values.tolist())
    with h5py.File(out_file, 'a') as hdf:
        for idx in np.arange(training_data.shape[0]):
            feature_gen_row_raw_validation_save(sample_rate, hdf, idx, training_data.iloc[idx])


def build_label_encoder():
    training_data = pd.read_csv(training_csv, header=0)
    y = np.array(training_data['name'].tolist())
    lb = LabelEncoder()
    lb.fit_transform(y)
    np.save('songs_training_data_classes.npy', lb.classes_)


# import librosa.display
# import matplotlib.pyplot as plt
#
# song_map_pairs = [
#     {'train': '/Users/hopper/Desktop/music_subset/Come as You Are.wma',
#      'validation': '/Users/hopper/Desktop/validation_music/Come as You Are.mp3'},
#     {'train': '/Users/hopper/Desktop/music_subset/Smells Like Teen Spirit.wma',
#      'validation': '/Users/hopper/Desktop/validation_music/Smells Like Teen Spirit.mp3'},
#     {'train': '/Users/hopper/Desktop/music_subset/Seven Nation Army.mp3',
#      'validation': '/Users/hopper/Desktop/validation_music/Seven Nation Army.mp3'},
# ]
#
# for song_map in song_map_pairs:
#     target_sample_rate = 44100
#     y_train, sample_rate = librosa.load(song_map['train'], sr=target_sample_rate, res_type='kaiser_fast')
#     y_validation, sample_rate = librosa.load(song_map['validation'], sr=target_sample_rate, res_type='kaiser_fast')
#     features_train = build_beat_audio_feature_sequences(y_train, target_sample_rate)
#     features_validation = build_beat_audio_feature_sequences(y_validation, target_sample_rate)
#     plt.figure()
#     ax1 = plt.subplot(2,1,1)
#     plt1 = librosa.display.specshow(features_validation.T, y_axis='chroma', x_axis='time')
#     plt.title('chroma_cq')
#     plt.colorbar()
#     plt.subplot(2,1,2, sharex=ax1)
#     plt2 = librosa.display.specshow(features_train.T, y_axis='chroma', x_axis='time')
#     plt.title('chroma_cens')
#     plt.colorbar()
#     plt.tight_layout()
#     plt.show()

def estimate_tempo(path):
    target_sample_rate = 44100
    y, sample_rate = librosa.load(path, sr=target_sample_rate)
    # y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=target_sample_rate
                                                 # , start_bpm=170
                                                 )
    print(str(tempo))


# estimate_tempo(root_music_dir + '/Highway to Hell.wma')
# estimate_tempo(root_music_validation_dir + '/Highway to Hell.mp3')
# estimate_tempo(root_music_dir + '/In Bloom.wma')
# estimate_tempo(root_music_validation_dir + '/In Bloom.mp3')
# estimate_tempo(root_music_dir + '/wonderwall.mp3')
# estimate_tempo(root_music_validation_dir + '/wonderwall.mp3')

# y_fast = librosa.effects.time_stretch(y, 2.0)
# tempo_fast, beat_frames_fast = librosa.beat.beat_track(y=y_fast, sr=target_sample_rate)
# tempo, beat_frames = librosa.beat.beat_track(y=y, sr=target_sample_rate)
# tempo

# path = root_music_dir + '/In Bloom.wma'
# y, sample_rate = librosa.load(path, sr=target_sample_rate)
#     y_harmonic, y_percussive = librosa.effects.hpss(y)
# tempo, beat_frames = librosa.beat.beat_track(y=y, sr=target_sample_rate
                                             # , start_bpm=70
                                             # )


# y, sample_rate = librosa.load('/Users/hopper/Desktop/music_subset/wonderwall.mp3', sr=target_sample_rate, res_type='kaiser_fast')
# raw_audio = build_audio_feature_sequences(X, target_sample_rate)

# write_training_csv()
# write_validation_csv()
# build_label_encoder()

# save_training_data_raw(44100, 'songs_training_data_44100_sequence_parts.h5')
# save_validation_data_raw(44100, 'songs_validation_data_44100_sequence_parts.h5')

# save_training_data_raw(44100, 'songs_training_data_44100_raw_parts.h5')

# save_training_data_raw(44100, 'songs_training_data_44100_raw_parts.h5')

# save_training_data_raw(44100, 'songs_training_data_44100_mfcc_parts.h5')
