"""Provides display options for audio files and their preprocessed features.

Note that the network does not use `librosa`_ anymore, because it has problems
with concurrent sample loading. This module has not been updated yet.

L8ER: Move away from librosa, use python_speech_features.

.. _librosa:
    https://librosa.github.io/librosa/index.html
"""

import os

import numpy as np
import librosa as rosa
from librosa import display
from matplotlib import pyplot as plt


DATASETS_PATH = '/home/marc/workspace/datasets/speech_data'


def display_sample_info(file_path, label=''):
    """Generate various representations a given audio file.
    E.g. Mel, MFCC and power spectrogram's.

    Args:
        file_path (str): Path to the audio file.
        label (str): Optional label to display for the given audio file.

    Returns:
        Nothing.
    """

    if not os.path.isfile(file_path):
        raise ValueError('{} does not exist.'.format(file_path))

    # By default, all audio is mixed to mono and resampled to 22050 Hz at load time.
    # y, sr = rosa.load(file_path, sr=None, mono=True)
    y, sr = rosa.load(file_path, sr=None, mono=True)

    # At 16000 Hz, 512 samples ~= 32ms. At 16000 Hz, 200 samples = 12ms. 16 samples = 1ms @ 16kHz.
    hop_length = 200    # Number of samples between successive frames e.g. columns if a spectrogram.
    f_max = sr / 2.     # Maximum frequency (Nyquist rate).
    f_min = 64.         # Minimum frequency.
    n_fft = 1024        # Number of samples in a frame.
    n_mfcc = 13         # Number of Mel cepstral coefficients to extract.
    n_mels = 80         # Number of Mel bins to generate
    win_length = 333    # Window length

    # Create info string.
    num_samples = y.shape[0]
    duration = rosa.get_duration(y=y, sr=sr)
    info_str = 'Label: {}\nPath: {}\nDuration={:.3f}s with {:,d} Samples\n' \
               'Sampling Rate={:,d} Hz\nMin, Max=[{:.2f}, {:.2f}]'
    info_str = info_str.format(label, file_path, duration, num_samples, sr, np.min(y), np.max(y))
    print(info_str)

    plt.figure()
    plt.subplot(3, 1, 1)
    display.waveplot(y, sr=sr)
    plt.title('Monophonic')

    # Plot waveforms.
    y_harm, y_perc = rosa.effects.hpss(y)
    plt.subplot(3, 1, 2)
    display.waveplot(y_harm, sr=sr, alpha=0.33)
    display.waveplot(y_perc, sr=sr, color='r', alpha=0.40)
    plt.title('Harmonic & Percussive')

    # Add file information.
    plt.subplot(3, 1, 3)
    plt.axis('off')
    plt.text(0.0, 1.0, info_str, color='black', verticalalignment='top')
    plt.tight_layout()

    # Calculating MEL spectrogram and MFCC.
    db_pow = np.abs(rosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)) ** 2

    s_mel = rosa.feature.melspectrogram(S=db_pow, sr=sr, hop_length=hop_length,
                                        fmax=f_max, fmin=f_min, n_mels=n_mels)

    s_mel = rosa.power_to_db(s_mel, ref=np.max)

    # Compute MFCC features from the MEL spectrogram.
    s_mfcc = rosa.feature.mfcc(S=s_mel, sr=sr, n_mfcc=n_mfcc)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    display.specshow(s_mfcc, sr=sr, x_axis='time', y_axis='linear', hop_length=hop_length)
    plt.set_cmap('magma')
    plt.xticks(rotation=295)
    plt.colorbar(format='%+2.0f')
    plt.title('MFCC')

    # And the first-order differences (delta features).
    mfcc_delta = rosa.feature.delta(s_mfcc, width=5, order=1)

    plt.subplot(1, 2, 2)
    display.specshow(mfcc_delta, sr=sr, x_axis='time', y_axis='linear', hop_length=hop_length)
    plt.set_cmap('magma')
    plt.xticks(rotation=295)
    plt.colorbar(format='%+2.0f')
    plt.title(r'$\Delta$ MFCC')
    plt.tight_layout()

    # STFT (Short-time Fourier Transform)
    # https://librosa.github.io/librosa/generated/librosa.core.stft.html
    plt.figure(figsize=(12, 10))
    db = rosa.amplitude_to_db(rosa.magphase(rosa.stft(y))[0], ref=np.max)
    plt.subplot(3, 2, 1)
    display.specshow(db, sr=sr, x_axis='time', y_axis='linear', hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')

    plt.subplot(3, 2, 2)
    display.specshow(db, sr=sr, x_axis='time', y_axis='log', hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')

    # CQT (Constant-T Transform)
    # https://librosa.github.io/librosa/generated/librosa.core.cqt.html
    cqt = rosa.amplitude_to_db(rosa.magphase(rosa.cqt(y, sr=sr))[0], ref=np.max)
    plt.subplot(3, 2, 3)
    display.specshow(cqt, sr=sr, x_axis='time', y_axis='cqt_note', hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrogram (note)')

    plt.subplot(3, 2, 4)
    display.specshow(cqt, sr=sr, x_axis='time', y_axis='cqt_hz', hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrogram (Hz)')

    plt.subplot(3, 2, 5)
    display.specshow(db, sr=sr, x_axis='time', y_axis='log', hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log power spectrogram')

    plt.subplot(3, 2, 6)
    display.specshow(s_mel, x_axis='time', y_axis='mel', hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    _test_txt_path = os.path.join('/home/marc/workspace/speech/data', 'train.txt')

    # Display specific sample info's.
    with open(_test_txt_path, 'r') as f:
        _lines = f.readlines()
        _line = _lines[0]
        _wav_path, txt = _line.split(' ', 1)
        _wav_path = os.path.join('/home/marc/workspace/datasets/speech_data', _wav_path)
        _txt = txt.strip()

        display_sample_info(_wav_path, label=_txt)
