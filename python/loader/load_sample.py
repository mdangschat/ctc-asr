"""Helper methods to load audio files."""

from os import path
import numpy as np
import scipy.io.wavfile as wav
import python_speech_features as psf

from python.params import FLAGS, NP_FLOAT


NUM_MFCC = 13           # Number of MFCC features to extract.
__WIN_STEP = 0.0125     # The step between successive windows in seconds.

# Mean and standard deviation values for normalization, according to `audio_set_info.py`.
__mean = [16.132103, -3.6725218, -6.214568, 3.6258953, -5.182402, -6.5299315, -7.6537876,
          -7.588856, -2.4014165, -0.76039016, -1.9804145, -1.3311814, -2.3036666, -0.0026071146,
          -0.002590179, 0.00046233408, -0.001012096, 0.0015143902, 0.006454269, 0.002413451,
          0.003715786, 0.0010370067, 0.001035958, 0.0012486908, 0.0006853765, 0.0014591864]
__GLOBAL_MEAN = np.array(__mean, dtype=NP_FLOAT).reshape([1, NUM_MFCC * 2])
__std = [3.0267246, 15.91595, 12.820962, 14.812205, 13.836283, 13.642111, 13.663327, 13.3519,
         12.499178, 12.373922, 11.249535, 10.477264, 9.673518, 0.62283206, 3.8496592, 3.2401998,
         3.3933964, 3.4643414, 3.610391, 3.605239, 3.6947887, 3.5006487, 3.472887, 3.197436,
         3.0140827, 2.8059297]
__GLOBAL_STD = np.array(__std, dtype=NP_FLOAT).reshape([1, NUM_MFCC * 2])


def load_sample(file_path, normalize='global'):
    """Loads the wave file and converts it into feature vectors.

    Args:
        file_path (str or bytes):
            A TensorFlow queue of file names to read from.
            `tf.py_func` converts the provided Tensor into `np.ndarray`s bytes.
        normalize (str or None):
            Whether to normalize the generated features or not. Supported types are:

                'global': Uses global mean and standard deviation values from `train.txt`.
                The normalization is being applied element wise.
                ([sample] - [mean]^T) / [std]^T
                Where brackets denote matrices or vectors.

                'local': Uses only the mean and standard deviation of the current sample.
                The normalization is being applied by ([sample] - mean_scalar) / std_scalar

                None: No normalization.

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

    if len(y) < 401:
        raise RuntimeError('Sample length () to short: {}'.format(len(y), file_path))

    if not sr == FLAGS.sampling_rate:
        raise RuntimeError('Sampling rate of {} found, expected {}.'
                           .format(sr, FLAGS.sampling_rate))

    # At 16000 Hz, 512 samples ~= 32ms. At 16000 Hz, 200 samples = 12ms. 16 samples = 1ms @ 16kHz.
    win_len = 0.025      # Window length in ms.
    win_step = __WIN_STEP  # Number of milliseconds between successive frames.
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
    if normalize is None:
        pass
    elif normalize == 'global':
        # Option 'global' is applied element wise.
        sample = (sample - __GLOBAL_MEAN) / __GLOBAL_STD
    elif normalize == 'local':
        # Option 'local' uses scalar values.
        sample = (sample - np.mean(sample)) / np.std(sample)
    else:
        raise ValueError('Invalid normalization method "{}".'.format(normalize))

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

    # Load audio data from drive.
    (sr, y) = wav.read(file_path)

    # The /2 is because `load_sample` skips every 2nd frame.
    return np.array(int(len(y) / sr / __WIN_STEP) // 2, dtype=np.int32)
