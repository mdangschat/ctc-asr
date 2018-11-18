"""
Generate `train.csv`, `dev.csv`, and `test.csv` for the `LibriSpeech`_
and `TEDLIUMv2`_ and `TIMIT`_ and `TATOEBA`_ and `Common Voice`_ datasets.

The selected parts of various datasets are merged into combined files at the end.

Downloading all supported archives requires approximately 80GB of free disk space.
The extracted corpus requires about 125GB of free disk space.

Generated data format:
    `path/to/sample.wav transcription of the sample wave file<new_line>`

    The transcription is in lower case letters a-z with every word separated
    by a <space>. Punctuation is removed.

.. _COMMON_VOICE:
    https://voice.mozilla.org/en

.. _LibriSpeech:
    http://openslr.org/12

.. _TATOEBA:
    https://tatoeba.org/eng/downloads

.. _TEDLIUMv2:
    http://openslr.org/19

.. _TIMIT:
    https://catalog.ldc.upenn.edu/LDC93S1
"""

import csv
import json
import os

from python.dataset.common_voice_loader import common_voice_loader
from python.dataset.config import CSV_DIR, CSV_DELIMITER
from python.dataset.config import CSV_HEADER_PATH, CSV_HEADER_LABEL, CSV_HEADER_LENGTH
from python.dataset.csv_file_helper import sort_csv_by_seq_len
from python.dataset.libri_speech_loeader import libri_speech_loader
from python.dataset.tatoeba_loader import tatoeba_loader
from python.dataset.tedlium_loader import tedlium_loader
from python.dataset.timit_loader import timit_loader
from python.util.params_helper import CORPUS_JSON_PATH


def generate_dataset(keep_archives=True, use_timit=True):
    """
    Download and pre-process the corpus.

    Args:
        keep_archives (bool): Cache downloaded archive files?
        use_timit (bool): Include the TIMIT corpus? If `True` it needs to be placed in the
            `./data/corpus/TIMIT/` directory by hand.

    Returns:
        Nothing.
    """
    # Common Voice
    cv_train, cv_test, cv_dev = common_voice_loader(keep_archives)

    # Libri Speech ASR
    ls_train, ls_test, ls_dev = libri_speech_loader(keep_archives)

    # Tatoeba
    tatoeba_train = tatoeba_loader(keep_archives)

    # TEDLIUM
    ted_train, ted_test, ted_dev = tedlium_loader(keep_archives)

    # TIMIT
    if use_timit:
        timit_train, timit_test = timit_loader()
    else:
        timit_train = _ = ''

    # Assemble and merge CSV files.
    # Train
    train = [cv_train, ls_train, tatoeba_train, ted_train, timit_train]
    train_csv, train_len = __merge_csv_files(train, 'train')

    # Test
    test = [cv_test, ls_test]
    _, test_len = __merge_csv_files(test, 'test')

    # Dev
    dev = [ls_dev]
    _, dev_len = __merge_csv_files(dev, 'dev')

    # Sort train.csv file (SortaGrad).
    boundaries, train_len_seconds = sort_csv_by_seq_len(train_csv)

    # Write corpus metadata to JSON.
    store_corpus_json(train_len, test_len, dev_len, boundaries, train_len_seconds)


def __merge_csv_files(csv_files, target):
    """
    Merge a list of CSV files into a single target CSV file.

    Args:
        csv_files (List[str]): List of paths to dataset CSV files.
        target (str): 'test', 'dev', 'train'

    Returns:
        Tuple[str, int]: Path to the created CSV file and the number of examples in it.
    """
    if target not in ['test', 'dev', 'train']:
        raise ValueError('Invalid target.')

    csv_fieldnames = [CSV_HEADER_PATH, CSV_HEADER_LABEL, CSV_HEADER_LENGTH]
    buffer = []

    # Read and merge files.
    for csv_file in csv_files:
        if not (os.path.exists(csv_file) and os.path.isfile(csv_file)):
            raise ValueError('File does not exist: ', csv_file)

        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=CSV_DELIMITER, fieldnames=csv_fieldnames)

            # Serialize reader data and remove header.
            lines = list(reader)[1:]

            # Add CSV data to buffer.
            buffer.extend(lines)

    # Write data to target file.
    target_file = os.path.join(CSV_DIR, '{}.csv'.format(target))
    with open(target_file, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, delimiter=CSV_DELIMITER, fieldnames=csv_fieldnames)
        writer.writeheader()

        writer.writerows(buffer)

        print('Added {:,d} lines to: {}'.format(len(buffer), target_file))

    return target_file, len(buffer)


def store_corpus_json(train_size, test_size, dev_size, boundaries, train_length):
    """
    Store corpus metadata in `/python/data/corpus.json`.

    Args:
        train_size (int): Number of training examples.
        test_size (int): Number of test examples.
        dev_size (int): Number of dev/validation examples.
        boundaries (List[int]): Array containing the bucketing boundaries.
        train_length (float): Total length of the training dataset in seconds.

    Returns:
        Nothing.
    """
    with open(CORPUS_JSON_PATH, 'w', encoding='utf-8') as f:
        data = {
            'train_size': train_size,
            'test_size': test_size,
            'dev_size': dev_size,
            'boundaries': boundaries,
            'train_length': train_length
        }
        json.dump(data, f, indent=2)


# Generate data.
if __name__ == '__main__':
    print('Starting to generate corpus.')

    generate_dataset(keep_archives=True, use_timit=True)

    print('Done. Please verify that "data/cache" contains only data that you want to keep.')
