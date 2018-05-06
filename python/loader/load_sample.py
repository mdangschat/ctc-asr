"""Helper methods to load audio files."""

from os import path
import numpy as np
import scipy.io.wavfile as wav
import python_speech_features as psf

from python.params import FLAGS, NP_FLOAT


NUM_MFCC = 13
WIN_STEP = 0.0125


def load_sample(file_path):
    """Loads the wave file and converts it into feature vectors.

    Args:
        file_path:
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

    # Load the audio files sample rate (`sr`) and data (`y`)
    (sr, y) = wav.read(file_path)

    if not sr == FLAGS.sampling_rate:
        raise TypeError('Sampling rate of {} found, expected {}.'.format(sr, FLAGS.sampling_rate))

    # At 16000 Hz, 512 samples ~= 32ms. At 16000 Hz, 200 samples = 12ms. 16 samples = 1ms @ 16kHz.
    win_len = 0.025      # Window length in ms.
    win_step = WIN_STEP  # Number of milliseconds between successive frames.
    f_max = sr / 2.      # Maximum frequency (Nyquist rate).
    f_min = 64.          # Minimum frequency.
    n_fft = 512          # Number of samples in a frame.
    n_mfcc = NUM_MFCC    # Number of Mel cepstral coefficients to extract.
    n_filters = 26       # Number of Mel bins to generate.

    # Compute MFCC features from the mel spectrogram.
    mfcc = psf.mfcc(signal=y, samplerate=sr, winlen=win_len, winstep=win_step,
                    numcep=n_mfcc, nfilt=n_filters, nfft=n_fft,
                    lowfreq=f_min, highfreq=f_max,
                    preemph=0.97, ceplifter=22, appendEnergy=True)

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


def wav_length(file_path):
    """Return time units for a given audio file, corresponding to the number of units if
    the actual features from `load_sample()` where computed.

    Args:
        file_path: Audio file path.

    Returns:
        np.ndarray: Number of time steps of feature extracted audio file.
    """
    if type(file_path) is not str:
        file_path = str(file_path, 'utf-8')

    if not path.isfile(file_path):
        raise ValueError('"{}" does not exist.'.format(file_path))

    # By default, all audio is mixed to mono and resampled to 22050 Hz at load time.
    (sr, y) = wav.read(file_path)

    if not sr == FLAGS.sampling_rate:
        raise TypeError('Sampling rate of {} found, expected {}.'.format(sr, FLAGS.sampling_rate))

    (sr, y) = wav.read(file_path)

    win_step = WIN_STEP  # Number of milliseconds between successive frames.

    # The /2 is because `load_sample` skips every 2nd frame.
    return np.array(int(len(y) / sr / win_step) // 2, dtype=np.int32)
