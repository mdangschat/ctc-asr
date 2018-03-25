"""Testing environment for `librosa` functionality."""

import os
import numpy as np
import librosa as rosa
from librosa import display
from matplotlib import pyplot as plt


def _load_sample(file_path, label=''):
    # L8ER Documentation
    # TODO Plot x-axis labels.
    if not os.path.isfile(file_path):
        raise ValueError('{} does not exist.'.format(file_path))

    # Set the hop length.
    # At 22050 Hz, 512 samples ~= 23ms
    # At 16000 Hz, 512 samples ~= review ms
    hop_length = 512

    # By default, all audio is mixed to mono and resampled to 22050 Hz at load time.
    wav_raw, sampling_rate = rosa.load(file_path, sr=None, mono=True)

    duration = rosa.get_duration(y=wav_raw, sr=sampling_rate)
    print('Label="{}"\nPath={}\nDuration={:.1f}s\nSampling Rate={:,d}Hz'
          .format(label, file_path, duration, sampling_rate))

    plt.figure()
    plt.subplot(3, 1, 1)
    display.waveplot(wav_raw, sr=sampling_rate)
    plt.title('Monophonic')

    y_harm, y_perc = rosa.effects.hpss(wav_raw)
    plt.subplot(3, 1, 2)
    display.waveplot(y_harm, sr=sampling_rate, alpha=0.25)
    display.waveplot(y_perc, sr=sampling_rate, color='r', alpha=0.5)
    plt.title('Harmonic + Percussive')

    plt.subplot(3, 1, 3)
    plt.axis('off')
    plt.text(0.0, 1.0,
             'Label="{}"\nPath={}\nDuration={:.1f}s\nSampling Rate={:,d}Hz'
             .format(label, file_path, duration, sampling_rate),
             color='black', verticalalignment='top')
    plt.tight_layout()

    # Compute MFCC features from the raw signal.
    # https://librosa.github.io/librosa/generated/librosa.core.stft.html
    mfcc = rosa.feature.mfcc(y=wav_raw, sr=sampling_rate, hop_length=hop_length, n_mfcc=13)
    print('MFCC', mfcc.shape)

    # And the first-order differences (delta features).
    mfcc_delta = rosa.feature.delta(mfcc)
    print('MFCC Delta:', mfcc_delta.shape)

    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    display.specshow(mfcc, sr=sampling_rate, x_axis='time', y_axis='linear')
    plt.title('MFCC (log)')

    plt.subplot(1, 2, 2)
    display.specshow(mfcc, sr=sampling_rate, x_axis='time', y_axis='linear')
    plt.title('MFCC Delta')
    plt.tight_layout()

    # STFT (Short-time Fourier Transform)
    #
    plt.figure(figsize=(12, 8))
    db = rosa.amplitude_to_db(rosa.magphase(rosa.stft(wav_raw))[0], ref=np.max)
    plt.subplot(4, 2, 1)
    display.specshow(db, sr=sampling_rate, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')

    plt.subplot(4, 2, 2)
    display.specshow(db, sr=sampling_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')

    # CQT (Constant-T Transform)
    # https://librosa.github.io/librosa/generated/librosa.core.cqt.html
    cqt = rosa.amplitude_to_db(rosa.magphase(rosa.cqt(wav_raw, sr=sampling_rate))[0], ref=np.max)
    plt.subplot(4, 2, 3)
    display.specshow(cqt, sr=sampling_rate, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrogram (note)')

    plt.subplot(4, 2, 4)
    display.specshow(cqt, sr=sampling_rate, x_axis='time', y_axis='cqt_hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrogram (Hz)')

    # Chromagram
    chroma = rosa.feature.chroma_cqt(y=wav_raw, sr=sampling_rate)
    plt.subplot(4, 2, 5)
    display.specshow(chroma, sr=sampling_rate, x_axis='time', y_axis='chroma')
    plt.colorbar()
    plt.title('Chromagram')

    plt.subplot(4, 2, 6)
    display.specshow(chroma, cmap='gray_r', sr=sampling_rate, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear power spectrogram (grayscale)')

    plt.subplot(4, 2, 7)
    display.specshow(db, sr=sampling_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log power spectrogram')

    plt.subplot(4, 2, 8)
    t_gram = rosa.feature.tempogram(y=wav_raw, sr=sampling_rate)
    display.specshow(t_gram, sr=sampling_rate, x_axis='time', y_axis='tempo')
    plt.colorbar()
    plt.title('Tempogram')
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    _timit_base_path = '/home/marc/workspace/speech/data/'
    _test_txt_path = os.path.join(_timit_base_path, 'test.txt')
    with open(_test_txt_path, 'r') as f:
        lines = f.readlines()
        line = lines[0]
        wav_path, txt = line.split(' ', 1)
        wav_path = os.path.join(_timit_base_path, 'timit/TIMIT', wav_path)
        print(wav_path, txt)

    _load_sample(rosa.util.example_audio_file(), label='librosa.utils.example_audio_file()')
    # _load_sample(wav_path, label=txt)
