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
        # generate_list('tedlium', 'train', dry_run=__dry_run),
        # generate_list('timit', 'train', dry_run=__dry_run),
        # generate_list('libri_speech', 'train', dry_run=__dry_run),
        generate_list('common_voice', 'train', dry_run=__dry_run),
        # generate_list('tatoeba', 'train', dry_run=__dry_run),
    ]

    __dev = [
        # generate_list('tedlium', 'dev', dry_run=__dry_run),
        # generate_list('libri_speech', 'dev', dry_run=__dry_run),
        # generate_list('common_voice', 'dev', dry_run=__dry_run),
    ]

    __test = [
        # generate_list('tedlium', 'test', dry_run=__dry_run),
        # generate_list('timit', 'test', dry_run=__dry_run),
        # generate_list('libri_speech', 'test', dry_run=__dry_run),
        generate_list('common_voice', 'test', dry_run=__dry_run),
    ]

    if not __dry_run:
        _merge_txt_files(__test, 'test')
        _merge_txt_files(__dev, 'dev')
        _merge_txt_files(__train, 'train')

    print('Done.')
