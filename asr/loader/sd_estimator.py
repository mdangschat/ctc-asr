"""
Calculate mean and standard deviation for a given training txt file.
"""

import os
import sys
import random

from multiprocessing import Pool, Lock, cpu_count
import numpy as np
from tqdm import tqdm

from asr.load_sample import load_sample


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
        random.shuffle(lines)
        lines = lines[: 2 ** 15]

        # Setup threadpool.
        lock = Lock()
        features = []   # Output buffer.

        with Pool(processes=cpu_count()) as pool:
            for feature in tqdm(
                pool.imap_unordered(__stat_calculator, lines, chunksize=4),
                    desc='Reading audio samples', total=len(lines), file=sys.stdout,
                    unit='samples', dynamic_ncols=True):
                lock.acquire()
                features.append(feature)
                lock.release()

        # Reduce the [num_samples, time, num_features] to [total_time, num_features] array.
        features = np.concatenate(features)

        print('mean = {}'.format(np.mean(features)))
        print('std = {}'.format(np.std(features)))
        means = np.mean(features, axis=0)
        print('__mean = [' + ', '.join(map(str, means)) + ']')
        stds = np.std(features, axis=0)
        print('__std = [' + ', '.join(map(str, stds)) + ']')


def __stat_calculator(line):
    # Python multiprocessing helper method.
    wav_path, _ = line.split(' ', 1)
    wav_path = os.path.join(DATASETS_PATH, wav_path)

    feature, _ = load_sample(wav_path, feature_type='mel', feature_normalization='none')
    assert len(feature) > 1

    return feature


if __name__ == '__main__':
    # Path to `train.txt` file.
    _test_txt_path = os.path.join('./data', 'train.txt')

    # Display dataset stats.
    calculate_dataset_stats(_test_txt_path)
