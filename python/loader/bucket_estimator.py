"""
Estimate optimal bucket sizes for training, based on `train.txt` content.
"""

import sys
import os

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from python.loader.load_sample import load_sample
from python.utils.storage import delete_file_if_exists


# Path to train.txt file.
TRAIN_TXT_PATH = '/home/marc/workspace/speech/data/train.txt'
# Path to dataset collection folder.
DATASET_PATH = '/home/marc/workspace/datasets/speech_data/'


def estimate_bucket_sizes(num_buckets=32, max_length=2000):
    """Estimate optimal bucket sizes based on the samples in `train.txt` file.
    Results are printed out or plotted.
    Optional, if `max_length` is greater than `0`, audio examples with feature vectors longer than
    `max_length` are being removed from the .txt file.

    Args:
        num_buckets (int): Number of buckets.
            Note that TensorFlow bucketing adds a smallest and largest bucket to the list.
        max_length (int): Maximum feature vector length of a preprocessed audio example.
            Longer ones are being removed from the .txt file.
            Set to `0` to disable removal.

    Returns:
        Nothing.
    """
    with open(TRAIN_TXT_PATH, 'r') as f:
        lines = f.readlines()

    overlength_counter = 0
    lengths = []
    tmp_lines = []

    # Progressbar
    for line in tqdm(lines, desc='Reading audio files', total=len(lines), file=sys.stdout,
                     unit='files', dynamic_ncols=True):
        wav_path, label = line.split(' ', 1)
        wav_path = os.path.join(DATASET_PATH, wav_path)
        _, sample_len = load_sample(wav_path, feature_type='mel',
                                    normalize_features=False, normalize_signal=False)

        if max_length > 0:
            if sample_len < max_length:
                lengths.append(sample_len)
                tmp_lines.append(line)
            else:
                overlength_counter += 1

        else:
            lengths.append(sample_len)
            tmp_lines.append(line)

    print()  # Clear line from tqdm progressbar.

    # Write reduced data back to .txt file, if selected.
    if max_length > 0:
        print('{} examples have a length greater than {} and have been removed from .txt file.'
              .format(overlength_counter, max_length))

        delete_file_if_exists(TRAIN_TXT_PATH)
        with open(TRAIN_TXT_PATH, 'w') as f:
            f.writelines(tmp_lines)

    print('Evaluated {} examples.'.format(len(lengths)))
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
