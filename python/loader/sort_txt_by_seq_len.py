"""
Sort a train.txt like file by it's audio files sequence length.
"""

import os
import sys

from tqdm import tqdm

from python.util import storage
from python.loader.load_sample import load_sample


DATASETS_PATH = '/home/marc/workspace/datasets/speech_data'


def _sort_txt_by_seq_len(txt_path):
    """Sort a train.txt like file by it's audio files sequence length.

    Args:
        txt_path (str): Path to the `train.txt`.

    Returns:
        Nothing.
    """

    # Read train.txt file.
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        read_length = len(lines)

        buffer = []   # Output buffer.

        for line in tqdm(lines, desc='Reading audio samples', total=len(lines), file=sys.stdout,
                         unit='samples', dynamic_ncols=True):
            wav_path, label = line.split(' ', 1)

            length = int(load_sample(os.path.join(DATASETS_PATH, wav_path))[1])
            buffer.append((length, wav_path, label))

        # Sort.
        buffer = sorted(buffer, key=lambda x: x[0])

        # Remove seq_length.
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


if __name__ == '__main__':
    # Path to `train.txt` file.
    _txt_path = os.path.join('/home/marc/workspace/speech/data', 'train.txt')

    # Display dataset stats.
    _sort_txt_by_seq_len(_txt_path)
