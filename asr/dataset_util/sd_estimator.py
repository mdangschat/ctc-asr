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


__DATASETS_PATH = '../datasets/speech_data'
__FEATURE_TYPE = 'mel'


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
        lines = lines[: int(2.0e5)]      # To fit in RAM and not crash Numpy.

        # Setup thread pool.
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
        print()

        means = np.mean(features, axis=0)
        print('__global_mean = [' + ', '.join(map(str, means)) + ']')
        stds = np.std(features, axis=0)
        print('__global_std = [' + ', '.join(map(str, stds)) + ']')


def __stat_calculator(line):
    # Python multiprocessing helper method.
    wav_path, _ = line.split(' ', 1)
    wav_path = os.path.join(__DATASETS_PATH, wav_path)

    feature, _ = load_sample(wav_path, feature_type=__FEATURE_TYPE, feature_normalization='none')
    assert len(feature) > 1, 'Empty feature: {}'.format(wav_path)

    return feature


if __name__ == '__main__':
    # Path to `train.txt` file.
    _test_txt_path = os.path.join('./data', 'train.txt')

    # Display dataset stats.
    calculate_dataset_stats(_test_txt_path)
