"""Generate `train.txt` and `test.txt` for the `LibriSpeech`_ and
`TIMIT`_ data sets.
 Additionally some information about the data set can be printed out.

Generated data format:
    `path/to/sample.wav transcription of the sample wave file<new_line>`

    The transcription is in lower case letters a-z with every word separated
    by a <space>. Punctuation is removed.

.. _LibriSpeech:
    http://openslr.org/12

.. _TIMIT:
    https://vcs.zwuenf.org/agct_data/timit
"""

import sys
import os
import re

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from loader import utils
from loader.load_sample import load_sample_dummy


# Path to the LibriSpeech ASR data set.
LIBRI_SPEECH_PATH = '/home/marc/workspace/speech/data/libri_speech/LibriSpeech/'

# Path to the TIMIT data set.
TIMIT_PATH = '/home/marc/workspace/speech/data/timit/TIMIT/'

# Where to generate the .txt files, e.g. /home/user/../data/<target>.txt
TARGET_PATH = '/home/marc/workspace/speech/data/'


def generate_list(data_path, target, loader, additional_output=False, dry_run=False):
    """Generate *.txt files containing the audio path and the corresponding sentence.
    Generated files are being stored at `TARGET_PATH`.

    Return additional data set information, see below.

    Args:
        data_path (str):
            Path the data set. Must match the `loader`'s capabilities.
        target (str):
            'train' or 'test'
        loader (function):
        additional_output (bool): Optional, default False.
            Convert the audio data to features and extract additional information.
            Prints out optimal bucket sizes.
        dry_run (bool): Optional, default False.
            Dry run, do not create output.txt file.

    Returns:
        Nothing.
    """

    if target != 'test' and target != 'train':
        raise ValueError('"{}" is not a valid target.'.format(target))

    if not os.path.isdir(data_path):
        raise ValueError('"{}" is not a directory.'.format(data_path))

    print('Starting to generate {}.txt file.'.format(target))

    # RegEX filter pattern for valid characters.
    pattern = re.compile(r'[^a-z ]+')

    # Load the output string.
    output = loader(data_path, pattern)

    # Calculate additional information. Note: Time consuming.
    if additional_output:
        lengths = []
        # Progressbar
        for line in tqdm(output, desc='Reading audio files', total=len(output), file=sys.stdout,
                         unit='files', dynamic_ncols=True):
            wav_path = line.split(' ', 1)[0]
            sample_len = load_sample_dummy(wav_path)
            lengths.append(sample_len)
        print()     # Clear line from tqdm progressbar.

        lengths = np.array(lengths)
        lengths = np.sort(lengths)
        num_buckets = 20
        step = len(lengths) // num_buckets
        buckets = '['
        for i in range(step, len(lengths), step):
            buckets += '{}, '.format(lengths[i])
        buckets = buckets[: -2] + ']'
        print('Suggested buckets: ', buckets)

        # Plot histogram.
        plt.figure()
        plt.hist(lengths, bins='auto', facecolor='green', alpha=0.75)

        plt.title('{}: Sequence Length\'s Histogram'.format(target))
        plt.ylabel('Count')
        plt.xlabel('Length')
        plt.grid(True)
        plt.show()

    # Write list to .txt file.
    target_path = os.path.join(TARGET_PATH, '{}.txt'.format(target))
    print('> Writing {} lines of {} files to {}'.format(len(output), target, target_path))

    if not dry_run:
        # Delete the old file if it exists.
        utils.delete_file_if_exists(target_path)

        # Write data to the file.
        with open(target_path, 'w') as f:
            f.writelines(output)


def _libri_speech_loader(data_path, pattern):
    """Build the output string that can be written to the desired *.txt file.

    Note: Since the TIMIT data set is relatively small, both train
    and test data is being merged into one .txt file.

    Args:
        data_path (str): Base path of the data set.
        pattern (str): RegEx pattern that is used as whitelist for the written label texts.

    Returns:
        [str]: List containing the output string that can be written to *.txt file.
    """
    output = []

    for root, dirs, files in os.walk(data_path):
        if len(dirs) is 0:
            # Get list of `.trans.txt` files.
            trans_txt_files = [f for f in files if f.endswith('.trans.txt')]
            # Verify that a `*.trans.txt` file exists.
            assert len(trans_txt_files) is 1, 'No .tans.txt file found: {}'.format(trans_txt_files)

            # Absolute path.
            trans_txt_path = os.path.join(root, trans_txt_files[0])

            # Load `.trans.txt` contents.
            with open(trans_txt_path, 'r') as f:
                lines = f.readlines()

            # Sanitize lines.
            lines = [line.lower().strip().split(' ', 1) for line in lines]

            for file_id, txt in lines:
                path = os.path.join(root, '{}.wav'.format(file_id))
                assert os.path.isfile(path), '{} not found.'.format(path)

                txt = re.sub(pattern, '', txt).strip().replace('  ', ' ')
                output.append('{} {}\n'.format(path, txt.strip()))

    return output


def _timit_loader(data_path, pattern):
    """Build the output string that can be written to the desired *.txt file.

    Note: Since the TIMIT data set is relatively small, both train
    and test data is being merged into one .txt file.

    Args:
        data_path (str): Base path of the data set.
        pattern (str): RegEx pattern that is used as whitelist for the written label texts.

    Returns:
        [str]: List containing the output string that can be written to *.txt file.
    """

    def _timit_loader_helper(data_set_path, master_txt_path, _pattern):
        # Internal helper method to generate the TIMIT .txt file.
        if not os.path.isfile(master_txt_path):
            raise ValueError('"{}" is not a file.'.format(master_txt_path))

        with open(master_txt_path, 'r') as f:
            master_data = f.readlines()

        _output = []

        for line in master_data:
            wav_path, txt_path, _, _ = line.split(',')
            txt_path = os.path.join(data_set_path, txt_path)

            basename = os.path.basename(wav_path)
            if 'SA1.WAV' == basename or 'SA2.WAV' == basename:
                continue

            with open(txt_path, 'r') as f:
                txt = f.readlines()
                assert len(txt) == 1, 'Text file contains to many lines. ({})'.format(txt_path)
                txt = txt[0].split(' ', 2)[2]

                txt = txt.lower()
                txt = re.sub(pattern, '', txt).strip()
                txt = txt.replace('  ', ' ')

                wav_path = os.path.join(data_set_path, wav_path)

                _output.append('{} {}\n'.format(wav_path, txt))

        return _output

    train_txt_path = os.path.join(data_path, 'train_all.txt')
    test_txt_path = os.path.join(data_path, 'test_all.txt')

    output = _timit_loader_helper(data_path, train_txt_path, pattern)
    output += _timit_loader_helper(data_path, test_txt_path, pattern)

    # Remove new line (`\n`) from last entry.
    output[-1] = output[-1].strip()
    return output


if __name__ == '__main__':
    # train.txt
    generate_list(LIBRI_SPEECH_PATH, 'train', _libri_speech_loader,
                  additional_output=True, dry_run=False)

    # test.txt
    generate_list(TIMIT_PATH, 'test', _timit_loader,
                  additional_output=True, dry_run=False)
