"""
Estimate optimal bucket sizes for training, based on `train.txt` content.
"""

import sys

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from python.loader.load_sample import load_sample_dummy


TRAIN_TXT_PATH = '/home/marc/workspace/speech/data/train.txt'


def estimate_bucket_sizes(num_buckets=20):
    # Estimate optimal bucket sizes.

    with open(TRAIN_TXT_PATH, 'r') as f:
        lines = f.readlines()

    lengths = []

    # Progressbar
    for line in tqdm(lines, desc='Reading audio files', total=len(lines), file=sys.stdout,
                     unit='files', dynamic_ncols=True):
        wav_path = line.split(' ', 1)[0]
        sample_len = load_sample_dummy(wav_path)
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
