"""
Print out a length distribution for used WAV files.
"""

import os
import sys
import pickle

from multiprocessing import Pool, Lock, cpu_count
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile

from asr.util.matplotlib_helper import pyplot_display
from asr.params import MIN_EXAMPLE_LENGTH, MAX_EXAMPLE_LENGTH


__DATASETS_PATH = '../datasets/speech_data'


def calculate_dataset_stats(txt_path):
    """Gather mean and standard deviation values. Averaged for every file in the
    training txt data file.

    Args:
        txt_path (str): Path to the `train.txt`.

    Returns:
        Nothing.
    """
    # Check if results are buffered.
    tmp_path = '/tmp/sample_length_dump_{}.p'.format(os.path.split(txt_path)[1])
    if not (os.path.exists(tmp_path) and os.path.isfile(tmp_path)):
        sample_lengths = []  # Output buffer.
        sample_lengths_sec = []  # Output buffer.

        # Read train.txt file.
        with open(txt_path, 'r') as f:
            lines = f.readlines()

            # Setup threadpool.
            lock = Lock()
            with Pool(processes=cpu_count()) as pool:
                for length, length_sec in tqdm(
                    pool.imap_unordered(_stat_calculator, lines, chunksize=4),
                        desc='Reading audio samples', total=len(lines), file=sys.stdout,
                        unit='samples', dynamic_ncols=True):
                    lock.acquire()
                    sample_lengths.append(length)
                    sample_lengths_sec.append(length_sec)
                    lock.release()

            pickle.dump(sample_lengths_sec, open(tmp_path, 'wb'))
            print('Stored data to {}'.format(tmp_path))

            total_len = np.sum(sample_lengths_sec)
            print('Total sample length={:.3f}s (~{}h) of {}.'
                  .format(total_len, int(total_len / 60 / 60), txt_path))
            print('Mean sample length={:.0f} ({:.3f})s.'
                  .format(np.mean(sample_lengths), np.mean(sample_lengths_sec)))

    else:
        print('Loading stored dump from {}'.format(tmp_path))
        sample_lengths_sec = pickle.load(open(tmp_path, 'rb'))
        print(len(sample_lengths_sec), type(sample_lengths_sec), sample_lengths_sec[3:5])

    # Plot histogram of WAV length distribution.
    _plot_wav_lengths(sample_lengths_sec)

    print('Done.')


def _stat_calculator(line):
    # Python multiprocessing helper method.
    wav_path, _ = line.split(' ', 1)
    wav_path = os.path.join(__DATASETS_PATH, wav_path)

    if not os.path.isfile(wav_path):
        raise ValueError('"{}" does not exist.'.format(wav_path))

    # Load the audio files sample rate (`sr`) and data (`y`).
    (sr, y) = wavfile.read(wav_path)

    length = len(y)
    length_sec = length / sr

    if length_sec < MIN_EXAMPLE_LENGTH:
        print('WARN: Too short example found: ', line, length_sec)

    if length_sec > MAX_EXAMPLE_LENGTH:
        print('WARN: Overlong example found: ', line, length_sec)

    return length, length_sec


@pyplot_display
def _plot_wav_lengths(plt, sample_lengths_sec):
    # Create figure.
    fig = plt.figure(figsize=(4.4, 2.2))
    plt.hist(sample_lengths_sec, bins=50, facecolor='green', alpha=0.75, edgecolor='black',
             linewidth=0.9)
    # plt.yscale('log')
    # plt.title('Sample Length in Seconds')
    plt.ylabel('count')
    plt.xlabel('length in seconds')
    plt.grid(True)

    # Finish plot by tightening everything up.
    plt.tight_layout()

    return fig


if __name__ == '__main__':
    # Path to `train.txt` file.
    _txt_path = os.path.join('./data', 'train.txt')

    # Display dataset stats.
    calculate_dataset_stats(_txt_path)
