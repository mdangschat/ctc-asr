"""L8ER"""

# TODO: Write librosa wrappers with good documentation and explanations.

import os
import numpy as np
import librosa as rosa


def load_sample(file_path, label='', expected_sr=None):
    # L8ER Documentation
    # review: label needed?

    if not os.path.isfile(file_path):
        raise ValueError('{} does not exist.'.format(file_path))

    # Set the hop length.
    # At 22050 Hz, 512 samples ~= 23ms. At 16000 Hz, 512 samples ~= TODO ms.
    hop_length = 200

    # By default, all audio is mixed to mono and resampled to 22050 Hz at load time.
    # y, sr = rosa.load(file_path, sr=None, mono=True)
    y, sr = rosa.load(file_path, sr=None, mono=True)

    if expected_sr is not None:
        assert sr == expected_sr, 'Sample rate of {} does not match the required rate of {}.'\
            .format(sr, expected_sr)

    # Info string.
    num_samples = y.shape[0]
    duration = rosa.get_duration(y=y, sr=sr)
    info_str = 'Label="{}"\nPath={}\nDuration={:.1f}s with {:,d} Samples\n' \
               'Sampling Rate={:,d}Hz\nMin, Max=[{:.2f}, {:.2f}]'\
        .format(label, file_path, duration, num_samples, sr, np.min(y), np.max(y))
    print(info_str)

    db_pow = np.abs(rosa.stft(y=y, n_fft=1024, hop_length=hop_length, win_length=400)) ** 2

    s_mel = rosa.feature.melspectrogram(S=db_pow, sr=sr, hop_length=hop_length,
                                        fmax=sr / 2., fmin=64., n_mels=80)
    s_mel = rosa.power_to_db(s_mel, ref=np.max)

    # Compute MFCC features from the mel spectrogram.
    mfcc = rosa.feature.mfcc(S=s_mel, sr=sr, n_mfcc=13)

    # And the first-order differences (delta features).
    mfcc_delta = rosa.feature.delta(mfcc, width=5, order=1)

    return mfcc, mfcc_delta
