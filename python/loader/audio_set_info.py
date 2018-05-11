"""Testing environment for `librosa`_ functionality.
Provides display options for audio files and their preprocessed features.

L8ER: Update documentation.
L8ER: Move away from librosa, use python_speech_features.

.. _librosa:
    https://librosa.github.io/librosa/index.html
"""

import os
import sys
import numpy as np
import librosa as rosa
from librosa import display
from matplotlib import pyplot as plt
from tqdm import tqdm

from python.loader.load_sample import load_sample


DATASETS_PATH = '/home/marc/workspace/datasets/speech_data'


def calculate_dataset_stats(txt_path):
    # L8ER: Document

    # Read train.txt file.
    with open(txt_path, 'r') as f:
        lines = f.readlines()

        means, stds = [], []    # Output buffers

        for line in tqdm(lines, desc='Reading audio samples', total=len(lines), file=sys.stdout,
                         unit='samples', dynamic_ncols=True):
            wav_path, _ = line.split(' ', 1)

            _, _, _, mean, std = sample_info(os.path.join(DATASETS_PATH, wav_path))
            means.append(mean)
            stds.append(std)

        means = np.mean(np.array(means), axis=0)
        stds = np.mean(np.array(stds), axis=0)

        print()
        print('Mean: min={}; max={}; mean={}'
              .format(np.amin(means), np.amax(means), np.mean(means)))
        print('[' + ', '.join(map(str, means)) + ']')

        print()
        print('STDs: min={}; max={}; mean={}'
              .format(np.amin(stds), np.amax(stds), np.mean(stds)))
        print('[' + ', '.join(map(str, stds)) + ']')


def sample_info(file_path):
    """Load a given audio file and pre process it with `loader.load_sample.load_sample()`,
    then extract some additional statistics.

    Args:
        file_path: Audio file path.

    Returns:
        (int, int, int, [float], [float]):
            Preprocessed sample length, maximum MFCC value, minimum MFCC value.
             As well as  the mean feature value, and standard deviation, element wise per element
             of the feature vector.
    """
    mfcc, sample_len = load_sample(file_path, normalize=False)
    mean = np.mean(mfcc, axis=0)
    std = np.std(mfcc, axis=0)

    return sample_len, np.amax(mfcc), np.amin(mfcc), mean, std


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

    # Display specific sample infos.
    # with open(_test_txt_path, 'r') as f:
    #     _lines = f.readlines()
    #     _line = _lines[0]
    #     _wav_path, txt = _line.split(' ', 1)
    #     _wav_path = os.path.join('/home/marc/workspace/datasets/speech_data', _wav_path)
    #     _txt = txt.strip()
    #
    #     display_sample_info(_wav_path, label=_txt)

    # Display dataset stats.
    calculate_dataset_stats(_test_txt_path)
