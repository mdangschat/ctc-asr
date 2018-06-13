"""Helper methods to load audio files."""

import os

import numpy as np
import scipy.io.wavfile as wav
import python_speech_features as psf

from asr.params import FLAGS, NP_FLOAT


NUM_FEATURES = 80        # Number of features to extract.
WIN_STEP = 0.010         # The step between successive windows in seconds.

# Mean and standard deviation values for normalization, according to `sd_estimator.py`.
__global_mean_mel = [7.186329, 7.187535, 7.203892, 7.230647, 7.209903, 7.218501, 7.2798657,
                     7.301412, 7.3203526, 7.3498015, 7.391833, 7.4655147, 7.5505347, 7.611956,
                     7.767723, 7.7433615, 7.706068, 7.680025, 7.6396456, 7.6872716, 7.524437,
                     7.467571, 7.4918127, 7.452687, 7.4972367, 7.487993, 7.425824, 7.48579,
                     7.4922857, 7.4987955, 7.558553, 7.5559945, 7.564094, 7.5956283, 7.6265736,
                     7.6404686, 7.7090054, 7.756986, 7.6623106, 7.616915, 7.6238174, 7.6244926,
                     7.588886, 7.577156, 7.5886173, 7.6223893, 7.771224, 7.8124237, 7.7559943,
                     7.7634764, 7.754607, 7.7148247, 7.69516, 7.727078, 7.7581134, 7.737, 7.712792,
                     7.712206, 7.669529, 7.6389294, 7.680363, 7.6614966, 7.6337266, 7.6464844,
                     7.664051, 7.6835184, 7.6805396, 7.669501, 7.6748223, 7.668175, 7.650711,
                     7.6439595, 7.637157, 7.6469655, 7.648362, 7.642455, 7.6323266, 7.6133137,
                     7.5660934, 7.372304]
__global_std_mel = [3.8424075, 4.0324845, 4.153479, 4.362712, 4.397133, 4.516968, 4.656457,
                    4.6514163, 4.67314, 4.730734, 4.8067, 4.868448, 4.89744, 4.905552, 4.9379997,
                    4.8784313, 4.792561, 4.726198, 4.6450977, 4.6177087, 4.4860077, 4.4091406,
                    4.403492, 4.338211, 4.338773, 4.292193, 4.207648, 4.254586, 4.245524, 4.2425914,
                    4.2830215, 4.2822533, 4.260944, 4.328642, 4.390646, 4.412897, 4.464121,
                    4.500312, 4.455902, 4.4344964, 4.449681, 4.4566607, 4.44826, 4.4577703,
                    4.4758186, 4.5077004, 4.601203, 4.6282897, 4.62047, 4.637828, 4.643226,
                    4.615952, 4.60123, 4.6199145, 4.6274114, 4.6066117, 4.576536, 4.5520673,
                    4.4881845, 4.408347, 4.3536773, 4.280483, 4.2183266, 4.186413, 4.166967,
                    4.156827, 4.1387854, 4.1213546, 4.1050014, 4.079438, 4.0540633, 4.0362515,
                    4.025153, 4.0391693, 4.0550666, 4.056436, 4.046349, 4.0555816, 4.0270762,
                    3.7482464]

__global_mean_mel = np.array(__global_mean_mel, dtype=NP_FLOAT).reshape([1, NUM_FEATURES])
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


def load_sample(file_path, feature_type=None, feature_normalization=None):
    """Loads the wave file and converts it into feature vectors.

    Args:
        file_path (str or bytes):
            A TensorFlow queue of file names to read from.
            `tf.py_func` converts the provided Tensor into `np.ndarray`s bytes.

        feature_type (str): Optional. If `None` is provided, use `FLAGS.feature_type`.
            Type of features to generate. Options are 'mel' and 'mfcc'.

        feature_normalization (str): Optional.
            If `None` is provided, use `FLAGS.feature_normalization`.

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

            'none': No normalization is being applied.


    Returns:
        np.ndarray:
            2D array with [time, num_features] shape, containing float.
        np.ndarray:
            Array containing a single int32.
    """
    __supported_feature_types = ['mel', 'mfcc']
    __supported_feature_normalizations = ['none', 'global', 'local', 'local_scalar']

    feature_type = feature_type if feature_type is not None else FLAGS.feature_type
    feature_normalization = feature_normalization if feature_normalization is not None \
        else FLAGS.feature_normalization

    if feature_type not in __supported_feature_types:
        raise ValueError('Requested feature type of {} isn\'t supported.'
                         .format(feature_type))

    if feature_normalization not in __supported_feature_normalizations:
        raise ValueError('Requested feature normalization method {} is invalid.'
                         .format(feature_normalization))

    if type(file_path) is not str:
        file_path = str(file_path, 'utf-8')

    if not os.path.isfile(file_path):
        raise ValueError('"{}" does not exist.'.format(file_path))

    # Load the audio files sample rate (`sr`) and data (`y`).
    (sr, y) = wav.read(file_path)

    if len(y) < 401:
        raise RuntimeError('Sample length {:,d} to short: {}'.format(len(y), file_path))

    if not sr == FLAGS.sampling_rate:
        raise RuntimeError('Sampling rate is {:,d}, expected {:,d}.'
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
        raise ValueError('Unsupported feature type: {}'.format(feature_type))

    # Make sure that data type matches TensorFlow type.
    sample = sample.astype(NP_FLOAT)

    # Drop every 2nd time frame, if requested.
    if FLAGS.features_drop_every_second_frame:
        sample = sample[:: 2, :]

    # Get length of the sample.
    sample_len = np.array(sample.shape[0], dtype=np.int32)

    # Apply feature normalization.
    sample = _feature_normalization(sample, feature_normalization, feature_type)

    # sample = [time, NUM_FEATURES], sample_len: scalar
    return sample, sample_len


def _mfcc(y, sr, win_len, win_step, num_features, n_fft, f_min, f_max):
    """Convert a wav signal into Mel Frequency Cepstral Coefficients (MFCC).

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
        raise ValueError('num_features is not a multiple of 2.')

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


def _feature_normalization(features, method, feature_type):
    """Normalize the given feature vector `y`, with the stated normalization `method`.

    Args:
        features (np.ndarray): The signal array
        method (str): Normalization method.

            'global': Uses global mean and standard deviation values from `train.txt`.
                The normalization is being applied element wise.
                ([sample] - [mean]^T) / [std]^T
                Where brackets denote matrices or vectors.

            'local': Use local (in sample) mean and standard deviation values, and apply the
                normalization element wise, like in `global`.

            'local_scalar': Uses only the mean and standard deviation of the current sample.
                The normalization is being applied by ([sample] - mean_scalar) / std_scalar

            'none': No normalization is being applied.

        feature_type (str): Feature type, see `load_sample` for details.

    Returns:
        np.ndarray: The normalized feature vector.
    """
    if method == 'none':
        return features
    elif method == 'global':
        # Option 'global' is applied element wise.
        if feature_type == 'mel':
            global_mean = __global_mean_mel
            global_std = __global_std_mel
        elif feature_type == 'mfcc':
            global_mean = __global_mean_mfcc
            global_std = __global_std_mfcc
        else:
            raise ValueError('Unsupported global feature type: {}'.format(feature_type))
        return (features - global_mean) / global_std
    elif method == 'local':
        return (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    elif method == 'local_scalar':
        # Option 'local' uses scalar values.
        return (features - np.mean(features)) / np.std(features)
    else:
        raise ValueError('Invalid normalization method: {}'.format(method))
