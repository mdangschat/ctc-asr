"""
Estimate optimal bucket sizes for training, based on `train.txt` content.
"""

import sys
import os

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from python.loader.load_sample import wav_features_length


# Path to train.txt file.
TRAIN_TXT_PATH = '/home/marc/workspace/speech/data/train.txt'
# Path to dataset collection folder.
DATASET_PATH = '/home/marc/workspace/datasets/speech_data/'


def estimate_bucket_sizes(num_buckets=32):
    """Estimate optimal bucket sizes based on the samples in `train.txt` file.
    Results are printed out or plotted.

    Args:
        num_buckets (int): Number of buckets.
            Note that TensorFlow bucketing adds a smallest and largest bucket to the list.

    Returns:
        Nothing.
    """
    with open(TRAIN_TXT_PATH, 'r') as f:
        lines = f.readlines()

    lengths = []

    # Progressbar
    for line in tqdm(lines, desc='Reading audio files', total=len(lines), file=sys.stdout,
                     unit='files', dynamic_ncols=True):
        wav_path = line.split(' ', 1)[0]
        wav_path = os.path.join(DATASET_PATH, wav_path)
        sample_len = wav_features_length(wav_path)
        lengths.append(sample_len)
    print()  # Clear line from tqdm progressbar.

    lengths = np.array(lengths)
    lengths = np.sort(lengths)
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


if __name__ == '__main__':
    estimate_bucket_sizes()
