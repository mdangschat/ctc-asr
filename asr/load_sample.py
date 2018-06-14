"""Helper methods to load audio files."""

import os

import numpy as np
import scipy.io.wavfile as wav
import python_speech_features as psf

from asr.params import FLAGS, NP_FLOAT


NUM_FEATURES = 80        # Number of features to extract.
WIN_STEP = 0.010         # The step between successive windows in seconds.

# Mean and standard deviation values for normalization, according to `sd_estimator.py`.
__global_mean_mel = [4.7068086, 4.707746, 4.7460155, 4.796398, 4.754262, 4.7698865, 4.8762083,
                     4.9139132, 4.950521, 5.009756, 5.0857997, 5.2143216, 5.3617506, 5.4676414,
                     5.7134595, 5.6838107, 5.627451, 5.5896454, 5.5361238, 5.605092, 5.3464785,
                     5.243403, 5.268254, 5.199207, 5.2647305, 5.252362, 5.1641765, 5.2591105,
                     5.2704406, 5.2792845, 5.363098, 5.3525057, 5.358001, 5.4047327, 5.449724,
                     5.4677167, 5.563849, 5.631127, 5.4898863, 5.4172597, 5.4236403, 5.4173265,
                     5.3607836, 5.340553, 5.3588047, 5.413754, 5.636812, 5.699799, 5.6173983,
                     5.6233163, 5.6073256, 5.5440435, 5.514795, 5.561803, 5.6040616, 5.57249,
                     5.5351043, 5.535768, 5.4754024, 5.4360967, 5.502247, 5.4872575, 5.455337,
                     5.480831, 5.51632, 5.5493746, 5.553231, 5.5487075, 5.560457, 5.5548997,
                     5.530741, 5.52049, 5.514872, 5.5293202, 5.5268707, 5.51584, 5.496515,
                     5.4664793, 5.395795, 5.0943375]
__global_std_mel = [4.572057, 4.8705215, 5.040407, 5.4131813, 5.41154, 5.6102276, 5.872859,
                    5.8397117, 5.8465767, 5.8920455, 5.961831, 5.9852357, 5.9579663, 5.9183073,
                    5.9364214, 5.8709893, 5.773992, 5.681853, 5.5784407, 5.554912, 5.4316735,
                    5.381952, 5.423274, 5.3746142, 5.384365, 5.3271503, 5.2113495, 5.25933,
                    5.242517, 5.237323, 5.2944417, 5.315908, 5.353105, 5.4128923, 5.46187,
                    5.4923344, 5.556873, 5.6001215, 5.573069, 5.5712557, 5.6054473, 5.643401,
                    5.655944, 5.6757135, 5.6962366, 5.7173777, 5.7901244, 5.8054304, 5.8142223,
                    5.850515, 5.872278, 5.865016, 5.8528547, 5.868978, 5.87194, 5.8525567, 5.823638,
                    5.7931156, 5.716647, 5.6196284, 5.547406, 5.43943, 5.346733, 5.294688, 5.251579,
                    5.2271223, 5.193026, 5.155854, 5.1303225, 5.100227, 5.074907, 5.0568743,
                    5.0388284, 5.0531445, 5.0736346, 5.0784674, 5.075495, 5.0911713, 5.075426,
                    4.774771]

__global_mean_mel = np.array(__global_mean_mel, dtype=NP_FLOAT).reshape([1, NUM_FEATURES])
__global_std_mel = np.array(__global_std_mel, dtype=NP_FLOAT).reshape([1, NUM_FEATURES])

__global_mean_mfcc = [9.426997, -6.8890533, -11.138549, 6.713418, -8.066264, -11.952292, -10.379166,
                      -11.082658, -5.707828, 2.0113358, -3.1875327, -1.6270261, -3.252464,
                      0.2855784, -3.1801, -3.2864435, -4.231724, 0.16664778, -2.2565804, 0.48551142,
                      -0.29102108, 0.3479757, -0.20767006, -0.0047650323, 0.41028392, 1.2668115e-05,
                      0.45425746, -0.46284926, 2.3234372, 0.537449, 1.5025678, -0.4661066, 1.464639,
                      -0.079518765, 0.014443183, -0.27803612, 2.7146852, 0.5934389, 0.7914084,
                      -0.5652875, 0.003154813, 0.03524304, 0.005882114, 0.008584797, 0.0064319326,
                      0.017226089, 0.0045734835, 0.009601189, 0.0013000737, 0.0073577967,
                      -0.001397686, 0.0025308584, -0.0015920785, 0.005332124, -0.0019712225,
                      0.003247717, -0.0017427814, 0.0026048524, -0.0008959117, 0.0011457605,
                      -0.0007285225, 0.00093035237, -9.754917e-05, -0.00012385545, 0.00030610533,
                      -0.00066856475, 0.0003622371, -0.001567156, 0.0011749621, -0.0022485885,
                      0.0014869008, -0.001502654, 0.0012752977, -0.0014840433, 0.001065706,
                      -0.0013628021, 0.0012775176, -0.0011729792, 0.0007976862, -0.0007730004]
__global_std_mfcc = [6.423895, 28.341375, 23.49994, 25.168423, 24.147701, 25.804447, 25.809204,
                     24.916998, 23.227983, 23.326353, 20.848839, 20.024536, 19.332407, 17.626677,
                     16.744526, 13.950745, 12.166232, 10.483157, 8.693975, 6.5546875, 4.690948,
                     2.851804, 1.0854248, 0.6021682, 2.2090838, 3.6029243, 4.916904, 6.1335196,
                     7.00139, 7.7228966, 8.280172, 8.558856, 8.6733, 8.530543, 8.259763, 7.666469,
                     7.074797, 6.408102, 5.56965, 4.575379, 0.5600785, 5.6883016, 4.8254123,
                     4.734227, 4.9179606, 5.5834203, 5.501075, 5.69335, 5.4618397, 5.502242,
                     5.11014, 4.9929585, 4.79154, 4.530152, 4.2223, 3.6987653, 3.23478, 2.7590928,
                     2.2961047, 1.7793312, 1.2725512, 0.7852492, 0.29627037, 0.16525057, 0.6024011,
                     1.0154012, 1.3814065, 1.7022192, 1.9847171, 2.2180414, 2.351548, 2.4511323,
                     2.4997685, 2.4855797, 2.4192617, 2.2977695, 2.160939, 1.9405161, 1.688251,
                     1.406067]

__global_mean_mfcc = np.array(__global_mean_mfcc, dtype=NP_FLOAT).reshape([1, NUM_FEATURES])
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
