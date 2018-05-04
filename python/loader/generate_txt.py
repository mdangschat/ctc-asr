"""Generate `train.txt` and `test.txt` for the `LibriSpeech`_ and
`TEDLIUMv2`_ and `TIMIT`_ datasets.
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
"""

import sys
import os
import re

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile

from loader import utils
from loader.load_sample import load_sample_dummy


# Path to the LibriSpeech ASR dataset.
LIBRI_SPEECH_PATH = '/home/marc/workspace/speech/data/libri_speech/LibriSpeech/'

# Path to the TEDLIUMv2 dataset.
TEDLIUM_PATH = '/home/marc/workspace/speech/data/tedlium/'

# Path to the TIMIT dataset.
TIMIT_PATH = '/home/marc/workspace/speech/data/timit/TIMIT/'

# Where to generate the .txt files, e.g. /home/user/../data/<target>.txt
TXT_TARGET_PATH = '/home/marc/workspace/speech/data/'


def generate_list(dataset_path, dataset_name, target, additional_output=False, dry_run=False):
    """Generate *.txt files containing the audio path and the corresponding sentence.
    Generated files are being stored at `TXT_TARGET_PATH`.

    Return additional data set information, see below.

    Args:
        dataset_path (str):
            Path the data set. Must match the `loader`'s capabilities.
        dataset_name (str):
            Name of the dataset. Supported datasets:
            `timit`, `libri_speech`, tedlium`
        target (str):
            'train' or 'test'
        additional_output (bool): Optional, default False.
            Convert the audio data to features and extract additional information.
            Prints out optimal bucket sizes.
        dry_run (bool): Optional, default False.
            Dry run, do not create output.txt file.

    Returns:
        Nothing.
    """
    # Supported loaders.
    loaders = {
        'timit': _timit_loader,
        'libri_speech': _libri_speech_loader,
        'tedlium': _tedlium_loader
    }

    if dataset_name not in loaders:
        raise ValueError('"{}" is not a supported dataset.'.format(dataset_name))
    else:
        loader = loaders[dataset_name]

    if target != 'test' and target != 'train':
        raise ValueError('"{}" is not a valid target.'.format(target))

    if not os.path.isdir(dataset_path):
        raise ValueError('"{}" is not a directory.'.format(dataset_path))

    print('Starting to generate {}.txt file.'.format(target))

    # RegEX filter pattern for valid characters.
    pattern = re.compile(r'[^a-z ]+')

    # Load the output string.
    output = loader(dataset_path, target, pattern)

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
    target_path = os.path.join(TXT_TARGET_PATH, '{}.txt'.format(target))
    print('> Writing {} lines of {} files to {}'.format(len(output), target, target_path))

    if not dry_run:
        # Delete the old file if it exists.
        utils.delete_file_if_exists(target_path)

        # Write data to the file.
        with open(target_path, 'w') as f:
            f.writelines(output)


def _libri_speech_loader(data_path, target, pattern):
    """Build the output string that can be written to the desired *.txt file.

    Note: Since the TIMIT data set is relatively small, both train
    and test data is being merged into one .txt file.

    Args:
        data_path (str): Base path of the data set.
        target (str): 'train' or 'test'
        pattern (str): RegEx pattern that is used as whitelist for the written label texts.

    Returns:
        [str]: List containing the output string that can be written to *.txt file.
    """
    train_folders = ['train-clean-100', 'train-clean-360']
    test_folders = ['dev-clean', 'test-clean']
    folders = train_folders if target is 'train' else test_folders

    output = []
    for folder in [os.path.join(data_path, f) for f in folders]:
        for root, dirs, files in os.walk(folder):
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


def _tedlium_loader(data_path, target, pattern):
    # TODO: Documentation
    # Note: Since TEDLIUM data is one large .wav file per speaker, this method needs to create
    #       several smaller part files. This takes some time.
    target_folders = {
        'validate': 'dev',
        'test': 'test',
        'train': 'train'
    }
    target_folder = os.path.join(data_path, 'TEDLIUM_release2', target_folders[target], 'stm')

    # Flag that marks time segments that should be skipped.
    ignore_flag = 'ignore_time_segment_in_scoring'

    print('target_folder:', target_folder)
    files = os.listdir(target_folder)

    output = []

    for stm_file in files:
        stm_file_path = os.path.join(target_folder, stm_file)
        print('stm_file_path:', stm_file, stm_file_path)
        with open(stm_file_path, 'r') as f:
            lines = f.readlines()

            wav_path = os.path.join(data_path, 'TEDLIUM_release2', target_folders[target],
                                    'sph', '{}.wav'.format(os.path.splitext(stm_file)[0]))
            assert os.path.isfile(wav_path), '{} not found.'.format(wav_path)

            (sample_rate, wav_data) = wavfile.read(wav_path)
            # TODO Stopped here!

            for i, line in enumerate(lines):
                if ignore_flag in line:
                    continue

                print(line)

                format_pattern = re.compile(
                    r'\w+ [0-9] \w+ ([0-9]+\.[0-9]+) ([0-9]+\.[0-9]+) <[\w,]+> ([\w ]+)'
                )
                res = re.search(format_pattern, line)
                start_time = float(res.group(1))
                end_time = float(res.group(2))
                text = res.group(3)
                print('DEBUG:', start_time, end_time, text)

                # Sanitize lines.
                text = text.lower()
                # Remove ` '`. TEDLIUM transcribes `i'm` as `i 'm`.
                text = text.replace(" '", '')
                text = re.sub(pattern, '', text).replace('  ', ' ').strip()
                output.append('{} {}\n'.format(wav_path, text))

                print(i, wav_path, text)

                break

            print('\n====================================================\n')

        break

    return output


# noinspection PyUnusedLocal
def _timit_loader(data_path, target, pattern):
    """Build the output string that can be written to the desired *.txt file.

    Note: Since the TIMIT data set is relatively small, both train
    and test data is being merged into one .txt file.

    Args:
        data_path (str): Base path of the data set.
        target (str): Not used at the moment for TIMIT data set. Set to `None`.
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
    generate_list(TEDLIUM_PATH, 'tedlium', 'test', additional_output=False, dry_run=True)

    # test.txt
    # generate_list(LIBRI_SPEECH_PATH, 'test', _libri_speech_loader,
    #               additional_output=False, dry_run=True)
