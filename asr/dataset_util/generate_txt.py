"""Generate `train.txt`, `dev.txt`, and `test.txt` for the `LibriSpeech`_
and `TEDLIUMv2`_ and `TIMIT`_ and `TATOEBA`_ and `Common Voice`_ datasets.

The selected parts of various datasets are merged into combined files at
the end.

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

.. _COMMON_VOICE:
    https://voice.mozilla.org/en
"""

import os
import re

from asr.util import storage
from asr.dataset_util.tatoeba_loader import tatoeba_loader
from asr.dataset_util.timit_loader import timit_loader
from asr.dataset_util.libri_speech_loeader import libri_speech_loader
from asr.dataset_util.tedlium_loader import tedlium_loader
from asr.dataset_util.common_voice_loader import common_voice_loader


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
            Note that it converts e.g. MP3 files to WAV files, no matter `dry_run`.

    Returns:
        (str): Path to the created TXT file.
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

    target_txt_path = os.path.join(TXT_TARGET_PATH, '{}_{}.txt'.format(dataset_name, target))
    print('Starting to generate: {}'.format(os.path.basename(target_txt_path)))

    # Load the output string. Format ['/path/s.wav label text\n', ...]
    output = loader(target)

    # Remove illegal characters from labels.
    output = _remove_illegal_characters(output)

    # Filter out labels that are only shorter than 2 characters.
    output = list(filter(lambda x: len((x.split(' ', 1)[-1]).strip()) >= 2, output))

    # Write list to .txt file.
    print('> Writing {} lines of {} files to {}'.format(len(output), target, target_txt_path))
    if not dry_run:
        # Delete the old file if it exists.
        storage.delete_file_if_exists(target_txt_path)

        # Write data to the file.
        with open(target_txt_path, 'w') as f:
            f.writelines(output)

    return target_txt_path


def _remove_illegal_characters(lines):
    """Remove every not whitelisted character from a list of formatted lines.

    Args:
        lines (List[str]): List if lines in the format '/path/to/file.wav label text here'

    Returns:
        List[str]: Filtered list of lines.
    """
    result = []
    for line in lines:
        path, text = line.split(' ', 1)
        text = re.sub(__PATTERN, '', text.lower()).strip().replace('  ', ' ')
        result.append('{} {}\n'.format(path, text))
    return result


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
    target_file = os.path.join(TXT_TARGET_PATH, '{}.txt'.format(target))
    with open(target_file, 'w') as f:
        f.writelines(buffer)
        print('Added {:,d} lines to: {}'.format(len(buffer), target_file))


if __name__ == '__main__':
    __dry_run = False

    __train = [
        generate_list('tatoeba', 'train', dry_run=__dry_run),       # TODO: last
        generate_list('tedlium', 'train', dry_run=__dry_run),
        generate_list('timit', 'train', dry_run=__dry_run),
        generate_list('libri_speech', 'train', dry_run=__dry_run),
        generate_list('common_voice', 'train', dry_run=__dry_run),
    ]

    __dev = [
        # generate_list('tedlium', 'dev', dry_run=__dry_run),
        generate_list('libri_speech', 'dev', dry_run=__dry_run),
        # generate_list('common_voice', 'dev', dry_run=__dry_run),
    ]

    __test = [
        # generate_list('tedlium', 'test', dry_run=__dry_run),
        # generate_list('timit', 'test', dry_run=__dry_run),
        generate_list('libri_speech', 'test', dry_run=__dry_run),
        generate_list('common_voice', 'test', dry_run=__dry_run),
    ]

    if not __dry_run:
        _merge_txt_files(__test, 'test')
        _merge_txt_files(__dev, 'dev')
        _merge_txt_files(__train, 'train')

    print('Done.')
