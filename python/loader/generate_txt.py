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
from scipy.io import wavfile

from python.params import FLAGS
from python.loader import utils


# Path to the LibriSpeech ASR dataset.
LIBRI_SPEECH_PATH = '/home/marc/workspace/speech/data/libri_speech/LibriSpeech/'

# Path to the TEDLIUMv2 dataset.
TEDLIUM_PATH = '/home/marc/workspace/speech/data/tedlium/'

# Path to the TIMIT dataset.
TIMIT_PATH = '/home/marc/workspace/speech/data/timit/TIMIT/'

# Where to generate the .txt files, e.g. /home/user/../data/<target>.txt
TXT_TARGET_PATH = '/home/marc/workspace/speech/data/'


def generate_list(dataset_path, dataset_name, target, dry_run=False):
    """Generate *.txt files containing the audio path and the corresponding sentence.
    Generated files are being stored at `TXT_TARGET_PATH`.

    Return additional data set information, see below.

    Args:
        dataset_path (str):
            Path the data set. Must match the `loader`'s capabilities.
        dataset_name (str):
            Name of the dataset. Supported dataset's:
            `timit`, `libri_speech`, tedlium`
        target (str):
            'train' or 'test'
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

    if target != 'test' and target != 'train' and target != 'validate':
        raise ValueError('"{}" is not a valid target.'.format(target))

    if not os.path.isdir(dataset_path):
        raise ValueError('"{}" is not a directory.'.format(dataset_path))

    target_path = os.path.join(TXT_TARGET_PATH, '{}_{}.txt'.format(dataset_name, target))
    print('Starting to generate {}.txt file.'.format(target))

    # RegEX filter pattern for valid characters.
    pattern = re.compile(r'[^a-z ]+')

    # Load the output string.
    output = loader(dataset_path, target, pattern)

    # Write list to .txt file.
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
                assert len(trans_txt_files) == 1, 'No .tans.txt file found: {}'\
                    .format(trans_txt_files)

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

    def seconds_to_sample(seconds, sr=16000):
        return int(seconds * sr)

    def write_wav_part(data, path, start, end, sr=16000):
        assert 0. <= start < (len(wav_data) / sr)
        assert start < end <= (len(wav_data) / sr)

        # print('Saving {:12,d}({:6.2f}s) to {:12,d}({:6.2f}s) at: {}'
        #       .format(seconds_to_sample(start), start, seconds_to_sample(end), end, path))

        utils.delete_file_if_exists(path)
        wavfile.write(path, sr, data[seconds_to_sample(start): seconds_to_sample(end)])

    target_folders = {
        'validate': 'dev',
        'test': 'test',
        'train': 'train'
    }
    target_folder = os.path.join(data_path, 'TEDLIUM_release2', target_folders[target], 'stm')

    # Flag that marks time segments that should be skipped.
    ignore_flag = 'ignore_time_segment_in_scoring'

    # RegEx pattern to extract TEDLIUM's .stm information's.
    format_pattern = re.compile(
        r"[.\w]+ [0-9] [.\w]+ ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) <[\w,]+> ([\w ']+)")

    files = os.listdir(target_folder)

    output = []

    for stm_file in tqdm(files, desc='Reading audio files', total=len(files), file=sys.stdout,
                         unit='files', dynamic_ncols=True):
        if os.path.splitext(stm_file)[1] != '.stm':
            # This check is required, since there are swap files, etc. in the TEDLIUM dataset.
            print('Invalid .stm file found:', stm_file)
            continue

        stm_file_path = os.path.join(target_folder, stm_file)
        with open(stm_file_path, 'r') as f:
            lines = f.readlines()

            wav_path = os.path.join(data_path, 'TEDLIUM_release2', target_folders[target],
                                    'sph', '{}.wav'.format(os.path.splitext(stm_file)[0]))
            assert os.path.isfile(wav_path), '{} not found.'.format(wav_path)

            # Load the audio data, to later split it into a part per speech segment.
            (sampling_rate, wav_data) = wavfile.read(wav_path)
            assert sampling_rate == FLAGS.sampling_rate

            for i, line in enumerate(lines):
                if ignore_flag in line:
                    continue

                res = re.search(format_pattern, line)
                if res is None:
                    raise RuntimeError('TEDLIUM loader error in file {}\nLine: {}'
                                       .format(stm_file_path, line))

                start_time = float(res.group(1))
                end_time = float(res.group(2))
                text = res.group(3)

                # Create new partial .wav file.
                part_path = '{}_{}.wav'.format(wav_path[: -4], i)
                write_wav_part(wav_data, part_path, start_time, end_time)

                # Sanitize lines.
                text = text.lower()
                # Remove ` '`. TEDLIUM transcribes `i'm` as `i 'm`.
                text = text.replace(" '", '')
                text = re.sub(pattern, '', text).replace('  ', ' ').strip()
                output.append('{} {}\n'.format(part_path, text))

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
    if target != 'test' and target != 'train':
        raise ValueError('Timit only supports `train` and `test` targets.')

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
    # TEDLIUM v2
    generate_list(TEDLIUM_PATH, 'tedlium', 'test', dry_run=False)
    generate_list(TEDLIUM_PATH, 'tedlium', 'validate', dry_run=False)
    generate_list(TEDLIUM_PATH, 'tedlium', 'train', dry_run=False)

    # TIMIT
    generate_list(TIMIT_PATH, 'timit', 'test', dry_run=False)
    generate_list(TIMIT_PATH, 'timit', 'train', dry_run=False)

    # LibriSpeech ASR Corpus
    generate_list(LIBRI_SPEECH_PATH, 'libri_speech', 'test', dry_run=False)
    generate_list(LIBRI_SPEECH_PATH, 'libri_speech', 'validate', dry_run=False)
    generate_list(LIBRI_SPEECH_PATH, 'libri_speech', 'train', dry_run=False)
