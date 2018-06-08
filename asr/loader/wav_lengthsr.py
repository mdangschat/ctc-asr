"""
Print out a length distribution for used WAV files.
"""

import os
import sys
import random

from multiprocessing import Pool, Lock, cpu_count
import numpy as np
from tqdm import tqdm
import scipy.io.wavfile as wav
# import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt


DATASETS_PATH = '../datasets/speech_data'


def calculate_dataset_stats(txt_path):
    """Gather mean and standard deviation values. Averaged for every file in the
    training txt data file.

    Args:
        txt_path (str): Path to the `train.txt`.

    Returns:
        Nothing.
    """
    # Read train.txt file.
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)

        # Setup threadpool.
        num_processes = cpu_count()
        lock = Lock()
        sample_lengths = []         # Output buffer.
        sample_lengths_sec = []     # Output buffer.

        with Pool(processes=num_processes) as pool:
            for length, length_sec in tqdm(
                pool.imap_unordered(__stat_calculator, lines, chunksize=4),
                    desc='Reading audio samples', total=len(lines), file=sys.stdout,
                    unit='samples', dynamic_ncols=True):
                lock.acquire()
                sample_lengths.append(length)
                sample_lengths_sec.append(length_sec)
                lock.release()

        print('mean sample length={:.3f} ({:.3f})s.'
              .format(np.mean(sample_lengths), np.mean(sample_lengths_sec)))

    # Plot histogram of WAV length distribution.
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(sample_lengths, bins='auto', facecolor='green', alpha=0.75)
    plt.title('Sample Length\'s Histogram')
    plt.ylabel('Count')
    plt.xlabel('Length')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(sample_lengths_sec, bins='auto', facecolor='green', alpha=0.75)
    plt.title('Sample Length in Seconds\'s Histogram')
    plt.ylabel('Count')
    plt.xlabel('Length in Seconds')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def __stat_calculator(line):
    # Python multiprocessing helper method.
    wav_path, _ = line.split(' ', 1)
    wav_path = os.path.join(DATASETS_PATH, wav_path)

    if not os.path.isfile(wav_path):
        raise ValueError('"{}" does not exist.'.format(wav_path))

    # Load the audio files sample rate (`sr`) and data (`y`).
    (sr, y) = wav.read(wav_path)

    length = len(y)
    return length, length / sr


if __name__ == '__main__':
    # Path to `train.txt` file.
    _test_txt_path = os.path.join('./data', 'train.txt')

    # Display dataset stats.
    calculate_dataset_stats(_test_txt_path)
