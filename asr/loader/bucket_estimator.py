"""
Estimate optimal bucket sizes for training, based on `train.txt` content.
"""

import sys
import os
import random

from multiprocessing import Pool, Lock
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from asr.loader.load_sample import load_sample


# Path to train.txt file.
TRAIN_TXT_PATH = './data/train.txt'
# Path to dataset collection folder.
DATASET_PATH = '../datasets/speech_data/'


def estimate_bucket_sizes(num_buckets=64):
    """Estimate optimal bucket sizes based on the samples in `train.txt` file.
    Results are printed out or plotted.
    Optional, if `max_length` is greater than `0`, audio examples with feature vectors longer than
    `max_length` are being removed from the .txt file.

    Args:
        num_buckets (int): Number of buckets.
            Note that TensorFlow bucketing adds a smallest and largest bucket to the list.

    Returns:
        Nothing.
    """
    with open(TRAIN_TXT_PATH, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)   # To ge a more precise estimated duration.

    # Setup threadpool.
    num_processes = 8
    lock = Lock()
    sample_lengths = []     # Output buffer.

    with Pool(processes=num_processes) as pool:
        for sample_len in tqdm(
                pool.imap_unordered(__estimate_bucket_size, lines, chunksize=8),
                desc='Reading audio files', total=len(lines), file=sys.stdout,
                unit='files', dynamic_ncols=True):
            lock.acquire()
            sample_lengths.append(sample_len)
            lock.release()

    sample_lengths = np.array(sample_lengths)

    print('Evaluated {:,d} examples.'.format(len(sample_lengths)))

    lengths = np.sort(sample_lengths)
    step = len(lengths) // num_buckets
    buckets = '['
    for i in range(step, len(lengths), step):
        buckets += '{}, '.format(lengths[i])
    buckets = buckets[: -2] + ']'
    print('Suggested buckets: ', buckets)

    # Plot histogram.
    plt.figure()
    plt.hist(lengths, bins='auto', facecolor='green', alpha=0.75)

    plt.title('Sequence Length\'s Histogram')
    plt.ylabel('Count')
    plt.xlabel('Length')
    plt.grid(True)

    plt.show()


def __estimate_bucket_size(line):
    # Python multiprocessing helper method.
    wav_path, label = line.split(' ', 1)
    wav_path = os.path.join(DATASET_PATH, wav_path)

    _, sample_len = load_sample(wav_path, feature_type='mel',
                                normalize_features=False, normalize_signal=False)
    return sample_len


if __name__ == '__main__':
    estimate_bucket_sizes()
