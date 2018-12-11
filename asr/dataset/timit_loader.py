"""
Load the `TIMIT`_ dataset.

.. _TIMIT:
    https://catalog.ldc.upenn.edu/LDC93S1
"""

import os

from scipy.io import wavfile

from asr.dataset.config import CORPUS_DIR
from asr.dataset.config import CSV_HEADER_PATH, CSV_HEADER_LABEL, CSV_HEADER_LENGTH
from asr.dataset.csv_helper import generate_csv
from asr.params import MIN_EXAMPLE_LENGTH, MAX_EXAMPLE_LENGTH

# Path to the TIMIT dataset.
__NAME = 'timit'
__FOLDER_NAME = 'timit/TIMIT'
__TARGET_PATH = os.path.realpath(os.path.join(CORPUS_DIR, __FOLDER_NAME))


def timit_loader():
    """
    Build all possible CSV files (e.g. `<dataset_name>_train.csv`, `<dataset_name>_test.csv`).

    Returns:
        List[str]: List containing the created CSV file paths.
    """

    targets = ['train', 'test']

    csv_paths = []
    for target in targets:
        # Generate the path and label for the `<target>.csv` file.
        output = __timit_loader(target)
        # Generate the `<target>.csv` file.
        csv_paths.append(generate_csv(__NAME, target, output))

    return csv_paths


def __timit_loader(target):
    """
    Build the data that can be written to the desired CSV file.

    Args:
        target (str): 'train' or 'test'.

    Returns:
        List[Dict]: List containing the dictionary entries for the timit_<target>.csv file.
    """
    if not os.path.isdir(__TARGET_PATH):
        raise ValueError('"{}" is not a directory.'.format(__TARGET_PATH))

    if target not in ('test', 'train'):
        raise ValueError('Timit only supports "train" and "test" targets.')

    # Select target. Location of TIMIT intern .txt file listings.
    if target == 'train':
        master_txt_path = os.path.join(__TARGET_PATH, 'train_all.txt')
    else:
        master_txt_path = os.path.join(__TARGET_PATH, 'test_all.txt')

    if not os.path.isfile(master_txt_path):
        raise ValueError('"{}" is not a file.'.format(master_txt_path))

    with open(master_txt_path, 'r') as file_handle:
        master_data = file_handle.readlines()

    output = []
    for line in master_data:
        wav_path, txt_path, _, _ = line.split(',')
        txt_path = os.path.join(__TARGET_PATH, txt_path)

        # Skip SAx.WAV files, since they are repeated by every speaker in the dataset.
        basename = os.path.basename(wav_path)
        if basename in ('SA1.WAV', 'SA2.WAV'):
            continue

        with open(txt_path, 'r') as file_handle:
            txt = file_handle.readlines()
            assert len(txt) == 1, 'Text file contains to many lines. ({})'.format(txt_path)
            txt = txt[0].split(' ', 2)[2]

            # Absolute path.
            wav_path = os.path.join(__TARGET_PATH, wav_path)

            # Validate that the example length is within boundaries.
            (sampling_rate, audio_data) = wavfile.read(wav_path)
            length_sec = len(audio_data) / sampling_rate
            if not MIN_EXAMPLE_LENGTH <= length_sec <= MAX_EXAMPLE_LENGTH:
                continue

            # Relative path to `DATASET_PATH`.
            wav_path = os.path.relpath(wav_path, CORPUS_DIR)

            output.append({
                CSV_HEADER_PATH: wav_path,
                CSV_HEADER_LABEL: txt.strip(),
                CSV_HEADER_LENGTH: length_sec
            })

    return output
