"""Helper methods to load audio files."""

import os

import numpy as np
import python_speech_features as psf
from scipy.io import wavfile

from python.params import FLAGS, NP_FLOAT

NUM_FEATURES = 80  # Number of features to extract.
WIN_STEP = 0.010  # The step between successive windows in seconds.

# Mean and standard deviation values for normalization, according to `sd_estimator.py`.
__global_mean_mel = [2.3643725, 2.3658302, 2.4267118, 2.493714, 2.4358225, 2.457301, 2.6084883,
                     2.6635098, 2.7155704, 2.8004122, 2.9063275, 3.0830536, 3.2855506, 3.433087,
                     3.7653785, 3.723953, 3.653416, 3.6144555, 3.5491524, 3.6348, 3.2863414,
                     3.1441562, 3.1720903, 3.0744805, 3.1635838, 3.148713, 3.0294576, 3.159464,
                     3.1758094, 3.19075, 3.302573, 3.2857845, 3.288685, 3.3494031, 3.4078734,
                     3.428585, 3.550336, 3.6342838, 3.4474418, 3.3498034, 3.353506, 3.3422055,
                     3.264557, 3.2368827, 3.260629, 3.3362062, 3.6275225, 3.7094772, 3.6026564,
                     3.6061692, 3.581369, 3.4969132, 3.4565856, 3.5157006, 3.5706284, 3.5330448,
                     3.4898846, 3.491176, 3.418615, 3.367653, 3.4527812, 3.4367719, 3.398318,
                     3.4368634, 3.4872725, 3.534044, 3.5417855, 3.5405827, 3.561789, 3.559142,
                     3.5270457, 3.5110383, 3.5041018, 3.5225167, 3.5185905, 3.5021145, 3.471864,
                     3.4285636, 3.3336356, 2.9349608]
__global_std_mel = [4.683658, 5.0328407, 5.3210506, 5.6568656, 5.645413, 5.979778, 6.1604786,
                    6.157099, 6.1572456, 6.1588545, 6.1628065, 6.163829, 6.1613555, 6.158614,
                    6.156115, 6.154111, 6.1517277, 6.09088, 6.011664, 5.989846, 5.91155, 5.8761873,
                    5.9272065, 5.896948, 5.9132185, 5.858601, 5.7479606, 5.792238, 5.776433,
                    5.771653, 5.828327, 5.857473, 5.909962, 5.960806, 5.9925447, 6.0153093,
                    6.0487833, 6.0672264, 6.065901, 6.0735846, 6.1037245, 6.13423, 6.1508107,
                    6.1522255, 6.1538477, 6.1544223, 6.1554217, 6.1550126, 6.157587, 6.1589856,
                    6.1597753, 6.1599255, 6.159658, 6.1595736, 6.1592717, 6.15908, 6.158359,
                    6.15591, 6.1524215, 6.102431, 6.0464783, 5.966331, 5.8915305, 5.8455663,
                    5.804497, 5.774609, 5.742994, 5.7113953, 5.695081, 5.6766944, 5.661086,
                    5.6510563, 5.6322803, 5.6332808, 5.641444, 5.6441584, 5.6373787, 5.6411486,
                    5.621142, 5.351847]

__global_mean_mel = np.array(__global_mean_mel, dtype=NP_FLOAT).reshape([1, NUM_FEATURES])
__global_std_mel = np.array(__global_std_mel, dtype=NP_FLOAT).reshape([1, NUM_FEATURES])

__global_mean_mfcc = [4.730534, -7.3349166, -9.780796, 6.9314823, -7.9121723, -10.662053,
                      -9.5768585, -10.032651, -5.5236244, 2.0479324, -3.1528459, -1.5990272,
                      -3.2561705, 0.275693, -3.093171, -3.2994611, -4.0499144, 0.1812272,
                      -2.2443013, 0.483652, -0.2815483, 0.3486155, -0.20616077, -0.0048009763,
                      0.40702823, 0.0116195865, 0.45759183, -0.45517603, 2.2553945, 0.5371715,
                      1.5070227, -0.4647816, 1.4569734, -0.07883959, 0.02163565, -0.2775471,
                      2.5048983, 0.60124767, 0.78632194, -0.55977076, 0.0031651238, 0.0354444,
                      0.006092494, 0.008659357, 0.006485829, 0.017330462, 0.0047889343, 0.009915208,
                      0.0014005781, 0.0077170786, -0.0012505432, 0.0024294439, -0.0014745407,
                      0.0054464587, -0.002083414, 0.0032398785, -0.0018625724, 0.00271708,
                      -0.0008366392, 0.0011059942, -0.0007035867, 0.00095390226, -9.8795565e-05,
                      -0.00013127626, 0.00031802547, -0.0007165586, 0.0003340503, -0.0015765982,
                      0.0012734793, -0.0022686757, 0.0014371471, -0.0014850029, 0.0013478879,
                      -0.0014295564, 0.0010426464, -0.0012896536, 0.0013519643, -0.0011751765,
                      0.0007979542, -0.00077622227]
__global_std_mfcc = [7.210424, 25.497797, 21.047031, 23.01523, 21.938654, 23.906382, 24.011173,
                     22.804012, 21.088669, 21.225285, 19.008026, 18.384405, 17.885141, 16.12674,
                     15.202616, 12.816007, 11.014082, 9.5317335, 7.883137, 6.0950375, 4.366502,
                     2.5781538, 0.98281187, 0.5564712, 2.0159588, 3.2917972, 4.5183554, 5.5584326,
                     6.429256, 6.990905, 7.486419, 7.7470737, 7.8522797, 7.710482, 7.451276,
                     6.934129, 6.471422, 5.909739, 5.028597, 4.2341876, 0.5296574, 5.423256,
                     4.504055, 4.4324713, 4.5735273, 5.13998, 5.0380945, 5.214711, 4.9902716,
                     5.0283475, 4.6835747, 4.589834, 4.440347, 4.1895604, 3.8285348, 3.3673792,
                     2.9969265, 2.4969265, 2.1295674, 1.6260618, 1.160426, 0.71743006, 0.27468532,
                     0.1498786, 0.55589545, 0.9169357, 1.2486045, 1.5712507, 1.7930905, 2.026393,
                     2.1862938, 2.2529569, 2.2875328, 2.2769501, 2.2300923, 2.1302288, 1.9534836,
                     1.7544352, 1.5622087, 1.269589]

__global_mean_mfcc = np.array(__global_mean_mfcc, dtype=NP_FLOAT).reshape([1, NUM_FEATURES])
__global_std_mfcc = np.array(__global_std_mfcc, dtype=NP_FLOAT).reshape([1, NUM_FEATURES])


def load_sample(file_path, feature_type=None, feature_normalization=None):
    """
    Loads the wave file and converts it into feature vectors.

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
    (sr, y) = wavfile.read(file_path)

    if len(y) < 401:
        raise RuntimeError('Sample length {:,d} to short: {}'.format(len(y), file_path))

    if not sr == FLAGS.sampling_rate:
        raise RuntimeError('Sampling rate is {:,d}, expected {:,d}.'
                           .format(sr, FLAGS.sampling_rate))

    # At 16000 Hz, 512 samples ~= 32ms. At 16000 Hz, 200 samples = 12ms. 16 samples = 1ms @ 16kHz.
    win_len = 0.025  # Window length in ms.
    win_step = WIN_STEP  # Number of milliseconds between successive frames.
    f_max = sr / 2.  # Maximum frequency (Nyquist rate).
    f_min = 64.  # Minimum frequency.
    n_fft = 1024  # Number of samples in a frame.

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
    """
    Convert a wav signal into Mel Frequency Cepstral Coefficients (MFCC).

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
    """
    Convert a wav signal into a logarithmically scaled mel filterbank.

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
    """
    Normalize the given feature vector `y`, with the stated normalization `method`.

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
