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

import json

from asr.dataset.common_voice_loader import common_voice_loader
from asr.dataset.csv_file_helper import sort_by_seq_len, get_corpus_length, merge_csv_files
from asr.dataset.libri_speech_loeader import libri_speech_loader
from asr.dataset.tatoeba_loader import tatoeba_loader
from asr.dataset.tedlium_loader import tedlium_loader
from asr.dataset.timit_loader import timit_loader
from asr.util.params_helper import CORPUS_JSON_PATH


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
    train_csv = merge_csv_files(
        [cv_train, ls_train, tatoeba_train, ted_train, timit_train],
        'train'
    )

    # Test
    test_csv = merge_csv_files(
        [cv_test, ls_test],
        'test'
    )

    # Dev
    dev_csv = merge_csv_files(
        [ls_dev],
        'dev'
    )

    # Sort train.csv file (SortaGrad).
    boundaries = sort_by_seq_len(train_csv)

    # Determine number of data entries and length in seconds per corpus.
    train_len, train_total_length_seconds = get_corpus_length(train_csv)
    test_len, _ = get_corpus_length(test_csv)
    dev_len, _ = get_corpus_length(dev_csv)

    # Write corpus metadata to JSON.
    store_corpus_json(train_len, test_len, dev_len, boundaries, train_total_length_seconds)


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
