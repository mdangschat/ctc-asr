"""
Calculate mean and standard deviation for a given training txt file.
"""

import os
import sys
import random

import numpy as np
from tqdm import tqdm

from python.loader.load_sample import load_sample


DATASETS_PATH = '/home/marc/workspace/datasets/speech_data'


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
        lines = lines[: 2 ** 16]

        features = []   # Output buffer.

        for line in tqdm(lines, desc='Reading audio samples', total=len(lines), file=sys.stdout,
                         unit='samples', dynamic_ncols=True):
            wav_path, _ = line.split(' ', 1)

            feature, _ = load_sample(os.path.join(DATASETS_PATH, wav_path),
                                     normalize_features=False, normalize_signal=False)
            features.append(feature)

        # Reduce the [num_samples, time, num_features] to [total_time, num_features] array.
        features = np.concatenate(features)

        print('mean = {}'.format(np.mean(features)))
        means = np.mean(features, axis=0)
        print('mean_vector = [' + ', '.join(map(str, means)) + ']')
        print('SD = {}'.format(np.std(features)))
        stds = np.std(features, axis=0)
        print('SD_vector = [' + ', '.join(map(str, stds)) + ']')


if __name__ == '__main__':
    # Path to `train.txt` file.
    _test_txt_path = os.path.join('/home/marc/workspace/speech/data', 'train.txt')

    # Display dataset stats.
    calculate_dataset_stats(_test_txt_path)
