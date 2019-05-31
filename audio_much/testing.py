import click
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.models import load_model
from audio_much.core import build_beat_audio_feature_sequences
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
import collections

encoder = LabelEncoder()
encoder.classes_ = np.load('songs_training_data_classes.npy')

lstm_model = load_model('model_raw_22050_lstm_02.h5')


def do_prediction_lstm(lstm_model, song_path, target_sample_rate=44100):
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


@click.command()
@click.argument('song_path')
def main(song_path):
    do_prediction_lstm(lstm_model, song_path, 44100)


if __name__ == "__main__":
    main()
