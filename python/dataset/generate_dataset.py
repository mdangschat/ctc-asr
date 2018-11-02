"""Generate `train.txt`, `dev.txt`, and `test.txt` for the `LibriSpeech`_
and `TEDLIUMv2`_ and `TIMIT`_ and `TATOEBA`_ and `Common Voice`_ datasets.

The selected parts of various datasets are merged into combined files at the end.

TODO: Add approximate download size and approximate disk space for extraction and approximate disk
      space for `data/corpus`.

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

import os

from python.dataset.config import TXT_DIR
from python.dataset.common_voice_loader import common_voice_loader
from python.dataset.libri_speech_loeader import libri_speech_loader
from python.dataset.tatoeba_loader import tatoeba_loader
from python.dataset.tedlium_loader import tedlium_loader
from python.dataset.timit_loader import timit_loader
from python.dataset.sort_txt_by_seq_len import sort_txt_by_seq_len


def generate_dataset(keep_archives=True, use_timit=True):
    """Download and preprocess the corpus.

    Args:
        keep_archives (bool): Cache downloaded archive files?
        use_timit (bool): Include the TIMIT corpus? If `True` it needs to be placed in the
            `./data/corpus/TIMIT/` directory by hand.

    Returns:
        Nothing
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

    # Assemble and merge .txt files.
    # Train
    train = [cv_train, ls_train, tatoeba_train, ted_train, timit_train]
    train_txt = _merge_txt_files(train, 'train')

    # Test
    test = [cv_test, ls_test]
    _ = _merge_txt_files(test, 'test')

    # Dev
    dev = [ls_dev]
    _ = _merge_txt_files(dev, 'dev')

    # Sort train.txt file (SortaGrad).
    sort_txt_by_seq_len(train_txt)


def _merge_txt_files(txt_files, target):
    """Merge a list of TXT files into a single target TXT file.

    Args:
        txt_files (List[str]): List of paths to dataset TXT files.
        target (str): 'test', 'dev', 'train'

    Returns:
        str: Path to the created TXT file.
    """
    if target not in ['test', 'dev', 'train']:
        raise ValueError('Invalid target.')

    buffer = []

    # Read and merge files.
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            buffer.extend(f.readlines())

    # Write data to target file.
    target_file = os.path.join(TXT_DIR, '{}.txt'.format(target))
    with open(target_file, 'w') as f:
        f.writelines(buffer)
        print('Added {:,d} lines to: {}'.format(len(buffer), target_file))

    return target_file


# Generate data.
if __name__ == '__main__':
    print('Starting to generate dataset.')

    generate_dataset(keep_archives=True, use_timit=True)

    print('Done. Please verify that `./data/cache` only contains data that you want to keep.')
