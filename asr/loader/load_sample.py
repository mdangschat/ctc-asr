"""Helper methods to load audio files."""

import os

import numpy as np
import scipy.io.wavfile as wav
import python_speech_features as psf

from asr.params import FLAGS, NP_FLOAT


NUM_FEATURES = 80        # Number of features to extract.
WIN_STEP = 0.010         # The step between successive windows in seconds.

# Mean and standard deviation values for normalization, according to `sd_estimator.py`.
__global_mean_mel = [4.354047, 4.354047, 4.3540845, 4.357667, 4.3698125, 4.4108806, 4.519213,
                     4.5571256, 4.5708227, 4.596592, 4.662528, 4.7651396, 4.860329, 4.9197493,
                     5.123324, 5.1341095, 5.129811, 5.093716, 5.027148, 5.0872297, 4.8830786,
                     4.81384, 4.8546863, 4.801289, 4.844881, 4.8178473, 4.729732, 4.804655,
                     4.8109293, 4.81992, 4.9031763, 4.920241, 4.9492464, 4.9834576, 5.0192394,
                     5.0393963, 5.130794, 5.201703, 5.0825453, 5.035488, 5.0602694, 5.0724974,
                     5.02827, 5.019642, 5.0453043, 5.093096, 5.2972627, 5.329108, 5.229246,
                     5.2359824, 5.2348723, 5.1936707, 5.17486, 5.206833, 5.2506204, 5.2549605,
                     5.253676, 5.279374, 5.2496357, 5.215727, 5.2695584, 5.234716, 5.2048283,
                     5.210627, 5.2164817, 5.224245, 5.207457, 5.189774, 5.2034307, 5.208409,
                     5.203033, 5.197059, 5.168414, 5.155391, 5.1316094, 5.10731, 5.0856447,
                     5.056711, 5.0009956, 4.789524]
__global_mean_mel = np.array(__global_mean_mel, dtype=NP_FLOAT).reshape([1, NUM_FEATURES])

__global_std_mel = [4.4118104, 4.6819134, 5.0119696, 5.6311355, 5.5846186, 5.750833, 5.9200892,
                    5.9095454, 5.9132557, 5.9314632, 5.94792, 5.955757, 5.9522185, 5.944907,
                    5.9594345, 5.937715, 5.9078236, 5.8320265, 5.7236266, 5.70917, 5.555901,
                    5.4937725, 5.538739, 5.4828353, 5.511166, 5.4699163, 5.3483996, 5.4094996,
                    5.389975, 5.381761, 5.4518833, 5.4702673, 5.497952, 5.543741, 5.5929484,
                    5.6288753, 5.704468, 5.7620792, 5.73591, 5.7385755, 5.7853155, 5.831621,
                    5.845873, 5.8697524, 5.8914313, 5.9038706, 5.922362, 5.925191, 5.9213223,
                    5.9273477, 5.933243, 5.933196, 5.932488, 5.9368825, 5.9366508, 5.9287834,
                    5.9194775, 5.916744, 5.906551, 5.8675117, 5.828888, 5.747699, 5.6658664,
                    5.6084695, 5.5524697, 5.499492, 5.434917, 5.384767, 5.3613143, 5.3394246,
                    5.320138, 5.3153634, 5.306109, 5.3106437, 5.299732, 5.2754273, 5.251981,
                    5.228218, 5.175849, 4.9269977]
__global_std_mel = np.array(__global_std_mel, dtype=NP_FLOAT).reshape([1, NUM_FEATURES])

__global_mean_mfcc = [7.90269, 8.347645, 8.687303, 8.688381, 8.689522, 8.695327, 8.721574, 8.730205,
                      8.735179, 8.744601, 8.76325, 8.789834, 8.813761, 8.828225, 8.907452, 8.904969,
                      8.892738, 8.871041, 8.839604, 8.863538, 8.785289, 8.766214, 8.789648,
                      8.774258, 8.79664, 8.787885, 8.753931, 8.781537, 8.781603, 8.78361, 8.819151,
                      8.827548, 8.8446455, 8.861377, 8.878465, 8.890945, 8.937761, 8.974516,
                      8.921929, 8.90544, 8.921057, 8.932361, 8.916152, 8.915106, 8.925816, 8.944602,
                      9.035392, 9.047635, 9.002181, 9.009189, 9.013126, 8.996938, 8.987281,
                      9.003349, 9.023387, 9.020733, 9.016268, 9.028506, 9.011191, 8.995056,
                      9.019399, 8.995772, 8.972218, 8.970518, 8.965054, 8.960189, 8.943783,
                      8.926244, 8.926092, 8.925793, 8.92112, 8.921144, 8.908918, 8.905724, 8.897008,
                      8.886151, 8.877501, 8.861594, 8.835282, 8.743987]
__global_mean_mfcc = np.array(__global_mean_mfcc, dtype=NP_FLOAT).reshape([1, NUM_FEATURES])

__global_std_mfcc = [3.7248473, 3.896349, 4.0420074, 4.152798, 4.2421, 4.302436, 4.316062, 4.303225,
                     4.3068814, 4.3543, 4.4055305, 4.450271, 4.4681425, 4.4753227, 4.495215,
                     4.463119, 4.426936, 4.392131, 4.3346643, 4.2998867, 4.2238445, 4.1661296,
                     4.1237774, 4.0638056, 4.030586, 3.9931803, 3.9483964, 3.9642053, 3.953392,
                     3.9474087, 3.955063, 3.9459088, 3.8762052, 3.9265602, 3.9929614, 4.008022,
                     4.035595, 4.054039, 4.0141144, 3.9931548, 3.994897, 3.9855788, 3.977878,
                     3.989702, 4.0110974, 4.0403585, 4.115289, 4.131577, 4.101389, 4.089623,
                     4.081747, 4.058913, 4.0473475, 4.058931, 4.0617085, 4.052006, 4.0400367,
                     4.0367203, 4.011046, 3.971689, 3.938358, 3.9060004, 3.8866398, 3.8634145,
                     3.8432512, 3.8217015, 3.792036, 3.7842822, 3.7792816, 3.7692058, 3.7609134,
                     3.7449331, 3.733456, 3.72996, 3.7164054, 3.6907291, 3.669801, 3.6685576,
                     3.6363444, 3.5091598]
__global_std_mfcc = np.array(__global_std_mfcc, dtype=NP_FLOAT).reshape([1, NUM_FEATURES])


def load_sample(file_path, feature_type='mel', normalize_features='local'):
    """Loads the wave file and converts it into feature vectors.

    Args:
        file_path (str or bytes):
            A TensorFlow queue of file names to read from.
            `tf.py_func` converts the provided Tensor into `np.ndarray`s bytes.

        feature_type (str): Type of features to generate. Options are `'mel'` and `'mfcc'`.

        normalize_features (str or bool):
            Whether to normalize the generated features with the stated method or not.
            Please consult `sample_normalization` for a complete list of normalization methods.

            'global': Uses global mean and standard deviation values from `train.txt`.
                The normalization is being applied element wise.
                ([sample] - [mean]^T) / [std]^T
                Where brackets denote matrices or vectors.

            'local': Use local (in sample) mean and standard deviation values, and apply the
                normalization element wise, like in `global`.

            'local_scalar': Uses only the mean and standard deviation of the current sample.
                The normalization is being applied by ([sample] - mean_scalar) / std_scalar

            False: No normalization is being applied.

    Returns:
        np.ndarray:
            2D array with [time, num_features] shape, containing float.
        np.ndarray:
            Array containing a single int32.
    """
    if type(file_path) is not str:
        file_path = str(file_path, 'utf-8')

    if not os.path.isfile(file_path):
        raise ValueError('"{}" does not exist.'.format(file_path))

    # Load the audio files sample rate (`sr`) and data (`y`).
    (sr, y) = wav.read(file_path)

    if len(y) < 401:
        raise RuntimeError('Sample length () to short: {}'.format(len(y), file_path))

    if not sr == FLAGS.sampling_rate:
        raise RuntimeError('Sampling rate of {} found, expected {}.'
                           .format(sr, FLAGS.sampling_rate))

    # At 16000 Hz, 512 samples ~= 32ms. At 16000 Hz, 200 samples = 12ms. 16 samples = 1ms @ 16kHz.
    win_len = 0.025         # Window length in ms.
    win_step = WIN_STEP     # Number of milliseconds between successive frames.
    f_max = sr / 2.         # Maximum frequency (Nyquist rate).
    f_min = 64.             # Minimum frequency.
    n_fft = 1024            # Number of samples in a frame.

    if feature_type == 'mfcc':
        sample = _mfcc(y, sr, win_len, win_step, NUM_FEATURES, n_fft, f_min, f_max)
    elif feature_type == 'mel':
        sample = _mel(y, sr, win_len, win_step, NUM_FEATURES, n_fft, f_min, f_max)
    else:
        raise ValueError('Unsupported feature type "{}".'.format(feature_type))

    # Make sure that data type matches TensorFlow type.
    sample = sample.astype(NP_FLOAT)

    # Get length of the sample.
    sample_len = np.array(sample.shape[0], dtype=np.int32)

    # Sample normalization.
    __global_mean = __global_mean_mel if feature_type == 'mel' else __global_mean_mfcc
    __global_std = __global_std_mel if feature_type == 'mel' else __global_std_mfcc
    sample = _feature_normalization(sample, normalize_features,
                                    global_mean=__global_mean, global_std=__global_std)

    # sample = [time, NUM_FEATURES], sample_len: scalar
    return sample, sample_len


def _mfcc(y, sr, win_len, win_step, num_features, n_fft, f_min, f_max):
    """Convert a wav signal into Mel Frequency Cepstral Coefficients.

    Args:
        y (np.ndarray): Wav signal.
        sr (int):  Sampling rate.
        win_len (float): Window length in seconds.
        win_step (float): Window stride in seconds.
        num_features (int): Number of features to generate.
        n_fft (int): Number of Fast Fourier Transforms.
        f_min (float): Minimum frequency to consider.
        f_max (float): Maximum frequency to consider.

    Returns:
        np.ndarray: MFCC feature vectors. Shape: [time, num_features]
    """
    if num_features % 2 != 0:
        raise ValueError('num_features not a multiple of 2.')

    # Compute MFCC features.
    mfcc = psf.mfcc(signal=y, samplerate=sr, winlen=win_len, winstep=win_step,
                    numcep=num_features // 2, nfilt=num_features, nfft=n_fft,
                    lowfreq=f_min, highfreq=f_max,
                    preemph=0.97, ceplifter=22, appendEnergy=True)

    # And the first-order differences (delta features).
    mfcc_delta = psf.delta(mfcc, 2)

    # Combine MFCC with MFCC_delta
    return np.concatenate([mfcc, mfcc_delta], axis=1)


def _mel(y, sr, win_len, win_step, num_features, n_fft, f_min, f_max):
    """Convert a wav signal into a logarithmically scaled mel filterbank.

    Args:
        y (np.ndarray): Wav signal.
        sr (int):  Sampling rate.
        win_len (float): Window length in seconds.
        win_step (float): Window stride in seconds.
        num_features (int): Number of features to generate.
        n_fft (int): Number of Fast Fourier Transforms.
        f_min (float): Minimum frequency to consider.
        f_max (float): Maximum frequency to consider.

    Returns:
        np.ndarray: Mel-filterbank. Shape: [time, num_features]
    """
    mel = psf.logfbank(signal=y, samplerate=sr, winlen=win_len,
                       winstep=win_step, nfilt=num_features, nfft=n_fft,
                       lowfreq=f_min, highfreq=f_max, preemph=0.97)
    return mel


def _feature_normalization(features, method, global_mean=__global_mean_mel,
                           global_std=__global_std_mel):
    """Normalize the given feature vector `y`, with the stated normalization `method`.

    Args:
        features (np.ndarray): The signal array
        method (str or bool): Normalization method.

            'global': Uses global mean and standard deviation values from `train.txt`.
                The normalization is being applied element wise.
                ([sample] - [mean]^T) / [std]^T
                Where brackets denote matrices or vectors.

            'local': Use local (in sample) mean and standard deviation values, and apply the
                normalization element wise, like in `global`.

            'local_scalar': Uses only the mean and standard deviation of the current sample.
                The normalization is being applied by ([sample] - mean_scalar) / std_scalar

            False: No normalization is being applied.

        global_mean (np.ndarray): (Optional).
            1D vector containing the global mean values per feature vector element.

        global_std (np.ndarray): (Optional).
            1D vector containing the global standard deviation values per feature vector element.

    Returns:
        np.ndarray: The normalized feature vector.
    """
    if not method:
        return features
    elif method == 'global':
        # Option 'global' is applied element wise.
        return (features - global_mean) / global_std
    elif method == 'local':
        return (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    elif method == 'local_scalar':
        # Option 'local' uses scalar values.
        return (features - np.mean(features)) / np.std(features)
    else:
        raise ValueError('Invalid normalization method "{}".'.format(method))
