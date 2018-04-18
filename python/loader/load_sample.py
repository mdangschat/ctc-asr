"""Helper methods to load audio files."""

from os import path
import librosa
import numpy as np

from s_params import FLAGS, NP_FLOAT


NUM_MFCC = 13


def load_sample(file_path):
    """Loads the wave file and converts it into feature vectors.

    Args:
        file_path (bytes):
            A TensorFlow queue of file names to read from.
            `tf.py_func` converts the provided Tensor into `np.ndarray`s bytes.

    Returns:
        np.ndarray:
            2D array with [time, num_features] shape, containing float.
        np.ndarray:
            Array containing a single int32.
    """
    print('start:', path.basename(file_path))
    if type(file_path) is not str:
        file_path = str(file_path, 'utf-8')

    if not path.isfile(file_path):
        raise ValueError('"{}" does not exist.'.format(file_path))

    # By default, all audio is mixed to mono and resampled to 22050 Hz at load time.
    y, sr = librosa.load(file_path, sr=None, mono=True)

    if not sr == FLAGS.sampling_rate:
        raise TypeError('Sampling rate of {} found, expected {}.'.format(sr, FLAGS.sampling_rate))

    # At 16000 Hz, 512 samples ~= 32ms. At 16000 Hz, 200 samples = 12ms. 16 samples = 1ms @ 16kHz.
    hop_length = 200    # Number of samples between successive frames e.g. columns if a spectrogram.
    f_max = sr / 2.     # Maximum frequency (Nyquist rate).
    f_min = 64.         # Minimum frequency.
    n_fft = 1024        # Number of samples in a frame.
    n_mfcc = NUM_MFCC   # Number of Mel cepstral coefficients to extract.
    n_mels = 80         # Number of Mel bins to generate
    win_length = 333    # Window length

    db_pow = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length,
                                 win_length=win_length)) ** 2

    s_mel = librosa.feature.melspectrogram(S=db_pow, sr=sr, hop_length=hop_length,
                                           fmax=f_max, fmin=f_min, n_mels=n_mels)

    s_mel = librosa.power_to_db(s_mel, ref=np.max)

    # Compute MFCC features from the mel spectrogram.
    mfcc = librosa.feature.mfcc(S=s_mel, sr=sr, n_mfcc=n_mfcc)

    # And the first-order differences (delta features).
    mfcc_delta = librosa.feature.delta(mfcc, width=5, order=1)

    # Combine MFCC with MFCC_delta
    sample = np.concatenate([mfcc, mfcc_delta], axis=0)

    sample = sample.astype(NP_FLOAT)
    sample = np.swapaxes(sample, 0, 1)

    # Skip every 2nd time frame.
    sample = sample[::2, :]

    sample_len = np.array(sample.shape[0], dtype=np.int32)

    sample = (sample - np.mean(sample)) / np.std(sample)

    print('done:', path.basename(file_path), sample_len)
    # `sample` = [time, num_features], `sample_len`: scalar
    return sample, sample_len
