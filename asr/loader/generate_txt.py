"""Generate `train.txt` and `test.txt` for the `LibriSpeech`_ and
`TEDLIUMv2`_ and `TIMIT`_ and `TATOEBA`_ datasets.
 Additionally some information about the data set can be printed out.

Generated data format:
    `path/to/sample.wav transcription of the sample wave file<new_line>`

    The transcription is in lower case letters a-z with every word separated
    by a <space>. Punctuation is removed.

.. _LibriSpeech:
    http://openslr.org/12

.. _TEDLIUMv2:
    http://openslr.org/19

.. _TIMIT:
    https://vcs.zwuenf.org/agct_data/timit

.. _TATOEBA:
    https://tatoeba.org/eng/downloads
"""

import sys
import os
import re
import csv
import subprocess

from multiprocessing import Pool, Lock, cpu_count
from tqdm import tqdm
from scipy.io import wavfile

from asr.params import FLAGS
from asr.util import storage
from asr.loader.tatoeba_loader import tatoeba_loader
from asr.loader.timit_loader import timit_loader
from asr.loader.libri_speech_loeader import libri_speech_loader
from asr.loader.tedlium_loader import tedlium_loader


# Dataset base path.
DATASET_PATH = '../datasets/speech_data'

# Where to generate the .txt files, e.g. /home/user/../data/<target>.txt
TXT_TARGET_PATH = './data/'

# RegEX filter pattern for valid characters.
__PATTERN = re.compile(r'[^a-z ]+')


def generate_list(dataset_name, target, dry_run=False):
    """Generate *.txt files containing the audio path and the corresponding sentence.
    Generated files are being stored at `TXT_TARGET_PATH`.

    Return additional data set information, see below.

    Args:
        dataset_name (str):
            Name of the dataset. Supported dataset's:
            'timit', 'libri_speech', 'tedlium', 'tatoeba'
        target (str):
            'train', 'test', or 'dev'
        dry_run (bool):
            Optional, default False.
            Dry running does not create output.txt files.

    Returns:
        Nothing.
    """
    # Supported loaders.
    loaders = {
        'timit': timit_loader,
        'libri_speech': libri_speech_loader,
        'tedlium': tedlium_loader,
        'common_voice': common_voice_loader,
        'tatoeba': tatoeba_loader
    }

    if dataset_name not in loaders:
        raise ValueError('"{}" is not a supported dataset.'.format(dataset_name))
    else:
        loader = loaders[dataset_name]

    if target != 'test' and target != 'train' and target != 'dev':
        raise ValueError('"{}" is not a valid target.'.format(target))

    if not os.path.isdir(dataset_path):
        raise ValueError('"{}" is not a directory.'.format(dataset_path))

    target_path = os.path.join(TXT_TARGET_PATH, '{}_{}.txt'.format(dataset_name, target))
    print('Starting to generate: {}'.format(os.path.basename(target_path)))

    # Load the output string. Format ['/path/s.wav label text\n', ...]
    output = loader(dataset_path, target)

    # Remove illegal characters from labels.
    output = _remove_illegal_characters(output)

    # Filter out labels that are only shorter than 2 characters.
    output = list(filter(lambda x: len((x.split(' ', 1)[-1]).strip()) >= 2, output))

    # Write list to .txt file.
    print('> Writing {} lines of {} files to {}'.format(len(output), target, target_path))
    if not dry_run:
        # Delete the old file if it exists.
        storage.delete_file_if_exists(target_path)

        # Write data to the file.
        with open(target_path, 'w') as f:
            f.writelines(output)


def _remove_illegal_characters(output):
    result = []
    for line in output:
        path, text = line.split(' ', 1)
        text = re.sub(__PATTERN, '', text.lower()).strip().replace('  ', ' ')
        result.append('{} {}'.format(path, text))
    return result


if __name__ == '__main__':
    __dry_run = False

    # TEDLIUMv2
    # generate_list(TEDLIUM_PATH, 'tedlium', 'test', dry_run=__dry_run)
    # generate_list(TEDLIUM_PATH, 'tedlium', 'dev', dry_run=__dry_run)
    # generate_list(TEDLIUM_PATH, 'tedlium', 'train', dry_run=__dry_run)

    # TIMIT
    # generate_list(__TIMIT_PATH, 'timit', 'test', dry_run=__dry_run)
    # generate_list(__TIMIT_PATH, 'timit', 'train', dry_run=__dry_run)

    # LibriSpeech ASR Corpus
    # generate_list(LIBRI_SPEECH_PATH, 'libri_speech', 'test', dry_run=__dry_run)
    # generate_list(LIBRI_SPEECH_PATH, 'libri_speech', 'dev', dry_run=__dry_run)
    # generate_list(LIBRI_SPEECH_PATH, 'libri_speech', 'train', dry_run=__dry_run)

    # Mozilla Common Voice
    # generate_list(COMMON_VOICE_PATH, 'common_voice', 'test', dry_run=__dry_run)
    # generate_list(COMMON_VOICE_PATH, 'common_voice', 'dev', dry_run=__dry_run)
    # generate_list(COMMON_VOICE_PATH, 'common_voice', 'train', dry_run=__dry_run)

    # Tatoeba
    generate_list(TATOEBA_PATH, 'tatoeba', 'train', dry_run=__dry_run)
