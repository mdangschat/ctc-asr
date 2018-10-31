"""Generate `train.txt`, `dev.txt`, and `test.txt` for the `LibriSpeech`_
and `TEDLIUMv2`_ and `TIMIT`_ and `TATOEBA`_ and `Common Voice`_ datasets.

The selected parts of various datasets are merged into combined files at
the end.

TODO: Add approximate download size and approximate disk space for extraction and approximate disk
TODO: space for `data/corpus`.

Generated data format:
    `path/to/sample.wav transcription of the sample wave file<new_line>`

    The transcription is in lower case letters a-z with every word separated
    by a <space>. Punctuation is removed.

.. _LibriSpeech:
    http://openslr.org/12

.. _TEDLIUMv2:
    http://openslr.org/19

.. _TIMIT:
    https://catalog.ldc.upenn.edu/LDC93S1

.. _TATOEBA:
    https://tatoeba.org/eng/downloads

.. _COMMON_VOICE:
    https://voice.mozilla.org/en
"""

import os

from python.dataset.config import TXT_DIR
from python.dataset.common_voice_loader import common_voice_loader
from python.dataset.libri_speech_loeader import libri_speech_loader
# tatoeba
from python.dataset.tedlium_loader import tedlium_loader
from python.dataset.timit_loader import timit_loader


def _merge_txt_files(txt_files, target):
    """Merge a list of TXT files into a single target TXT file.

    Args:
        txt_files (List[str]): List of paths to dataset TXT files.
        target (str): 'test', 'dev', 'train'

    Returns:
        Nothing.
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


if __name__ == '__main__':
    # Generate data.
    # Common Voice
    cv_train, cv_test, cv_dev = common_voice_loader()

    # Libri Speech ASR
    ls_train, ls_test, ls_dev = libri_speech_loader()

    # Tatoeba
    # TODO

    # TEDLIUM
    ted_train, ted_test, ted_dev = tedlium_loader()

    # TIMIT
    timit_train, timit_test = timit_loader()

    # TODO Assemble and merge .txt files.
    # Train
    _merge_txt_files([cv_train, ted_train, timit_train], 'train')
    # Test
    _merge_txt_files([cv_test, ted_test], 'test')
    # Dev
    _merge_txt_files([cv_dev], 'dev')

    # TODO Sort train.txt file (SortaGrad).

    print('Done.')
    print('Please verify that ./data/cache only contains required data.')
