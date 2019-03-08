import os
import pandas as pd
import glob

import librosa
# import librosa.display
import numpy as np
import re
from functools import partial
from sklearn.preprocessing import LabelEncoder

# Do this before use matplotlib on Mac: https://stackoverflow.com/questions/29433824/unable-to-import-matplotlib-pyplot-as-plt-in-virtualenv
# import matplotlib.pyplot as plt

def write_training_csv():
    import csv
    with open('songs_training.csv', 'w') as out_file:
        writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(['name', 'location'])
        for root, directories, filenames in os.walk('/Users/hopper/Desktop/music'):
            for filename in filenames:
                if re.match('^.*\.(wma|WMA|mp3|MP3|wav|WAV|m4a|M4A)$', filename):
                    writer.writerow([filename, os.path.join(root, filename)])


def display_song(song_location):
    raw_tertiary, sampling_rate = librosa.load(song_location)
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(raw_tertiary, sr=sampling_rate)


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


def build_float_audio_segments(X):
    len_second = 1.0
    split = int(target_sample_rate * len_second)
    split_remainder = int(X.shape[0] % split)
    split_count = int(X[:-split_remainder].shape[0] / split)
    audio_segments = np.split(X[:-split_remainder], split_count)
    return [x.astype('float') for x in audio_segments]


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


def feature_gen_row_raw_save(target_sample_rate, out_file, idx, row):
    try:
        print('generating features for: ' + row['location'])
        X, sample_rate = librosa.load(row['location'], sr=target_sample_rate, res_type='kaiser_fast')
        X_50, sample_rate_50 = librosa.load(row['location'], sr=target_sample_rate, res_type='kaiser_fast', offset=0.5)
        # X_25, sample_rate_25 = librosa.load(row['location'], sr=target_sample_rate, res_type='kaiser_fast', offset=0.25)
        raw_audio = build_float_audio_segments(X)
        raw_audio_50 = build_float_audio_segments(X_50)
        # raw_audio_25 = build_float_audio_segments(X_25)
        raw_audio_full = raw_audio + raw_audio_50    # + raw_audio_25
        df = pd.DataFrame({'feature': pd.Series(raw_audio_full),
                           'label': pd.Series(([row['name']] * len(raw_audio_full)), dtype='str')
                           })
    except Exception as e:
        print('error while generating features for: ' + row['location'])
        print(e)
        raise e
        # df = pd.DataFrame({'feature': pd.Series(), 'label': pd.Series(dtype='str')})
    df.to_hdf(out_file, key=('df_' + str(idx)), mode='a')


def save_training_data(sample_rate, out_file):
    training_data = pd.read_csv('songs_training.csv', header=0)
    feature_gen_row_fn = partial(feature_gen_row, sample_rate)
    actual_training_data = training_data.apply(feature_gen_row_fn, axis=1)
    actual_training_data.columns = ['feature', 'label']
    df = pd.DataFrame(actual_training_data.values.tolist())
    df.columns = ['feature', 'label']
    df = df.dropna()
    df.to_pickle(out_file)


def save_training_data_raw(sample_rate, out_file):
    training_data = pd.read_csv('songs_training.csv', header=0)
    # actual_training_data = training_data.apply(feature_gen_row_fn, axis=1)
    # df = pd.concat(actual_training_data.values.tolist())
    for idx in np.arange(training_data.shape[0]):
        feature_gen_row_raw_save(sample_rate, out_file, idx, training_data.iloc[idx])


def build_label_encoder():
    training_data = pd.read_csv('songs_training.csv', header=0)
    y = np.array(training_data['name'].tolist())
    lb = LabelEncoder()
    lb.fit_transform(y)
    np.save('songs_training_data_classes.npy', lb.classes_)


# save_training_data_raw(22050, 'songs_training_data_22050_raw_parts.h5')

# save_training_data_raw(44100, 'songs_training_data_44100_raw_parts.h5')

# for col in df.columns:
#     print('col: ' + str(col) + ' dtype: ' + str(df[col].dtype))
#
# df.to_hdf('songs_training_data_22050_raw_small.h5', key='df', mode='w')
# df['feature'].convert_objects(convert_numeric=True)

# df['feature'].apply(lambda x: x[~np.isnan(x)])
#  df[(df.applymap(type) != df.iloc[0].apply(type)).any(axis=1)]
# df['feature'].apply(lamba x: x.dropna())
#
# np.vectorize(fn_thing)(np.array([1,2,3]))
#
# fn_thing = lambda x: type(x)
# fn_thing_first = lambda x: np.array([fn_thing(xi) for xi in x])[0]
#
# get_first_type_fn = lambda x: type(x[0])
#
# df[]
#
# df = save_training_data_raw(22050, 'songs_training_data_22050_raw.h5')
# df.to_pickle('songs_training_data_22050_raw.pickle')
# df.to_hdf('songs_training_data_22050_raw.h5', key='df', mode='w')
#
# group_size = 1
# rows = df.shape[0]
# group_count = np.ceil(rows / group_size)
# for group_idx in np.arange(group_count):
#     min_idx = int(group_size * group_idx)
#     max_idx = int(min_idx + group_size)
#     print('writing indices ' + str(min_idx) + ' through ' + str(max_idx))
#     df[min_idx:max_idx].to_hdf('songs_training_data_22050_raw_parts.h5', key = 'df_' + str(min_idx), append=True)
#
#
# for i in
# df.to_hdf('songs_training_data_22050_raw_parts.h5', key='df', mode='w', complevel=9)