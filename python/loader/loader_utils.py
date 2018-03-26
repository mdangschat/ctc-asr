"""Testing environment for `librosa` functionality."""

import os
import numpy as np
import librosa as rosa
from librosa import display
from matplotlib import pyplot as plt


def sample_info(file_path, label=''):
    # L8ER Documentation

    if not os.path.isfile(file_path):
        raise ValueError('{} does not exist.'.format(file_path))

    # Set the hop length.
    # At 22050 Hz, 512 samples ~= 23ms
    # At 16000 Hz, 512 samples ~= review ms
    hop_length = 200

    # By default, all audio is mixed to mono and resampled to 22050 Hz at load time.
    # y, sr = rosa.load(file_path, sr=None, mono=True)
    y, sr = rosa.load(file_path, sr=None, mono=True)

    # Create info string.
    num_samples = y.shape[0]
    duration = rosa.get_duration(y=y, sr=sr)
    info_str = 'Label="{}"\nPath={}\nDuration={:.1f}s with {:,d} Samples\n' \
               'Sampling Rate={:,d}Hz\nMin, Max=[{:.2f}, {:.2f}]'\
        .format(label, file_path, duration, num_samples, sr, np.min(y), np.max(y))

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
    plt.title('Harmonic + Percussive')

    # Add file information.
    plt.subplot(3, 1, 3)
    plt.axis('off')
    plt.text(0.0, 1.0, info_str, color='black', verticalalignment='top')
    plt.tight_layout()

    db_pow = np.abs(rosa.stft(y=y, n_fft=1024, hop_length=hop_length, win_length=400)) ** 2
    mel_spect = rosa.feature.melspectrogram(S=db_pow, sr=sr, hop_length=hop_length,
                                            fmax=sr / 2., fmin=64., n_mels=80)
    mel_spect = rosa.power_to_db(mel_spect, ref=np.max)

    # Compute MFCC features from the raw signal.
    mfcc = rosa.feature.mfcc(S=mel_spect, sr=sr, n_mfcc=13)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    display.specshow(mfcc, sr=sr, x_axis='time', y_axis='linear', hop_length=hop_length)
    plt.set_cmap('magma')
    plt.xticks(rotation=295)
    plt.colorbar(format='%+2.0f')
    plt.title('MFCC')

    # And the first-order differences (delta features).
    mfcc_delta = rosa.feature.delta(mfcc, width=5, order=1)

    plt.subplot(1, 2, 2)
    display.specshow(mfcc_delta, sr=sr, x_axis='time', y_axis='linear', hop_length=hop_length)
    plt.xticks(rotation=295)
    plt.colorbar(format='%+2.0f')
    plt.title(r'$\Delta$ MFCC')
    plt.tight_layout()

    # STFT (Short-time Fourier Transform)
    # https://librosa.github.io/librosa/generated/librosa.core.stft.html
    plt.figure(figsize=(12, 10))
    db = rosa.amplitude_to_db(rosa.magphase(rosa.stft(y))[0], ref=np.max)
    plt.subplot(3, 2, 1)
    display.specshow(db, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')

    plt.subplot(3, 2, 2)
    display.specshow(db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')

    # CQT (Constant-T Transform)
    # https://librosa.github.io/librosa/generated/librosa.core.cqt.html
    cqt = rosa.amplitude_to_db(rosa.magphase(rosa.cqt(y, sr=sr))[0], ref=np.max)
    plt.subplot(3, 2, 3)
    display.specshow(cqt, sr=sr, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrogram (note)')

    plt.subplot(3, 2, 4)
    display.specshow(cqt, sr=sr, x_axis='time', y_axis='cqt_hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrogram (Hz)')

    plt.subplot(3, 2, 5)
    display.specshow(db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log power spectrogram')

    plt.subplot(3, 2, 6)
    display.specshow(mel_spect, x_axis='time', y_axis='mel', hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    _timit_base_path = '/home/marc/workspace/speech/data/'
    _test_txt_path = os.path.join(_timit_base_path, 'train.txt')
    with open(_test_txt_path, 'r') as f:
        lines = f.readlines()
        line = lines[0]
        wav_path, txt = line.split(' ', 1)
        txt = txt.strip()
        wav_path = os.path.join(_timit_base_path, 'timit/TIMIT', wav_path)

    # _sample_info(rosa.util.example_audio_file(), label='librosa.utils.example_audio_file()')
    sample_info(wav_path, label=txt)
