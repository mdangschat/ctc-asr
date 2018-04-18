"""Helper methods to load audio files."""

from os import path
import scipy.io.wavfile as wav
import python_speech_features as psf
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
    if type(file_path) is not str:
        file_path = str(file_path, 'utf-8')

    if not path.isfile(file_path):
        raise ValueError('"{}" does not exist.'.format(file_path))

    # By default, all audio is mixed to mono and resampled to 22050 Hz at load time.
    (sr, y) = wav.read(file_path)

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

    # Compute MFCC features from the mel spectrogram.
    mfcc = psf.mfcc(y, sr)

    # And the first-order differences (delta features).
    mfcc_delta = psf.delta(mfcc, 2)

    # Combine MFCC with MFCC_delta
    sample = np.concatenate([mfcc, mfcc_delta], axis=1)

    # Data type.
    sample = sample.astype(NP_FLOAT)

    # Skip every 2nd time frame.
    sample = sample[:: 2, :]

    sample_len = np.array(sample.shape[0], dtype=np.int32)

    # Sample normalization.
    sample = (sample - np.mean(sample)) / np.std(sample)

    # `sample` = [time, num_features], `sample_len`: scalar
    return sample, sample_len
