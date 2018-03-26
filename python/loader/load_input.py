"""L8ER"""
# TODO: Write librosa wrappers with good documentation and explanations.

import os
import numpy as np
import librosa as rosa


def load_sample(file_path, expected_sr=None):
    # TODO: This method can be moved to s_input.py
    # L8ER Documentation

    if not os.path.isfile(file_path):
        raise ValueError('{} does not exist.'.format(file_path))

    # By default, all audio is mixed to mono and resampled to 22050 Hz at load time.
    # y, sr = rosa.load(file_path, sr=None, mono=True)
    y, sr = rosa.load(file_path, sr=None, mono=True)

    # Set generally used variables. TODO: Document their purpose.
    # At 22050 Hz, 512 samples ~= 23ms. At 16000 Hz, 512 samples ~= TODO ms.
    hop_length = 200
    f_max = sr / 2.
    f_min = 64.

    if expected_sr is not None:
        assert sr == expected_sr, 'Sample rate of {:,d} does not match the required rate of {:,d}.'\
            .format(sr, expected_sr)

    db_pow = np.abs(rosa.stft(y=y, n_fft=1024, hop_length=hop_length, win_length=400)) ** 2

    s_mel = rosa.feature.melspectrogram(S=db_pow, sr=sr, hop_length=hop_length,
                                        fmax=f_max, fmin=f_min, n_mels=80)

    s_mel = rosa.power_to_db(s_mel, ref=np.max)

    # Compute MFCC features from the mel spectrogram.
    mfcc = rosa.feature.mfcc(S=s_mel, sr=sr, n_mfcc=13)

    # And the first-order differences (delta features).
    # mfcc_delta = rosa.feature.delta(mfcc, width=5, order=1)

    print('mfcc:', mfcc, type(mfcc))    # review
    return mfcc
