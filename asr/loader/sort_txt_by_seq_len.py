"""
Sort a train.txt like file by it's audio files sequence length.
"""

import os
import sys

from multiprocessing import Pool, Lock, cpu_count
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from asr.util import storage
from asr.load_sample import load_sample


DATASETS_PATH = '../datasets/speech_data'


def _sort_txt_by_seq_len(txt_path, num_buckets=64, max_length=1700):
    """Sort a train.txt like file by it's audio files sequence length.
    Additionally outputs longer than `max_length` are being discarded from the given TXT file.
    Also it prints out optimal bucket sizes after computation.

    Args:
        txt_path (str): Path to the `train.txt`.
        num_buckets (int): Number ob buckets to split the input into.
        max_length (int): Positive integer. Max length for a feature vector to keep.
            Set to `0` to keep everything.

    Returns:
        Nothing.
    """
    # Read train.txt file.
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        read_length = len(lines)

        # Setup threadpool.
        lock = Lock()
        buffer = []   # Output buffer.

        with Pool(processes=cpu_count()) as pool:
            for result in tqdm(pool.imap_unordered(__feature_length, lines, chunksize=4),
                               desc='Reading audio samples', total=len(lines), file=sys.stdout,
                               unit='samples', dynamic_ncols=True):
                lock.acquire()
                buffer.append(result)
                lock.release()

        # Sort by sequence length.
        buffer = sorted(buffer, key=lambda x: x[0])

        # Remove samples longer than `max_length` points.
        if max_length > 0:
            original_length = len(buffer)
            buffer = [s for s in buffer if s[0] < 1750]
            print('Removed {:,d} samples from training.'.format(original_length - len(buffer)))

        # Calculate optimal bucket sizes.
        lengths = [l[0] for l in buffer]
        step = len(lengths) // num_buckets
        buckets = '['
        for i in range(step, len(lengths), step):
            buckets += '{}, '.format(lengths[i])
        buckets = buckets[: -2] + ']'
        print('Suggested buckets: ', buckets)

        # Plot histogram of feature vector length distribution.
        plt.figure()
        plt.hist(lengths, bins='auto', facecolor='green', alpha=0.75)
        plt.title('Sequence Length\'s Histogram')
        plt.ylabel('Count')
        plt.xlabel('Length')
        plt.grid(True)
        plt.show()

        # Remove sequence length.
        buffer = ['{} {}'.format(p, l) for _, p, l in buffer]

    # Write back to file.
    assert read_length == len(buffer)
    storage.delete_file_if_exists(txt_path)
    with open(txt_path, 'w') as f:
        f.writelines(buffer)

    with open(txt_path, 'r') as f:
        assert len(f.readlines()) == read_length, \
            'Something went wrong writing the data back to file.'
        print('Successfully sorted {} lines of {}'.format(read_length, txt_path))


def __feature_length(line):
    # Python multiprocessing helper method.
    wav_path, label = line.split(' ', 1)
    length = int(load_sample(os.path.join(DATASETS_PATH, wav_path))[1])
    return length, wav_path, label


if __name__ == '__main__':
    # Path to `train.txt` file.
    _txt_path = os.path.join('./data', 'train.txt')

    # Display dataset stats.
    _sort_txt_by_seq_len(_txt_path)
