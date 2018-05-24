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
import csv
import subprocess

from tqdm import tqdm
from scipy.io import wavfile

from asr.params import FLAGS
from asr.util import storage


# Dataset base path.
DATASET_PATH = '../datasets/speech_data'

# Path to the LibriSpeech ASR dataset.
LIBRI_SPEECH_PATH = os.path.join(DATASET_PATH, 'libri_speech/LibriSpeech')
# Path to the TEDLIUMv2 dataset.
TEDLIUM_PATH = os.path.join(DATASET_PATH, 'tedlium')
# Path to the TIMIT dataset.
TIMIT_PATH = os.path.join(DATASET_PATH, 'timit/TIMIT')
# Path to the Mozilla Common Voice dataset.
COMMON_VOICE_PATH = os.path.join(DATASET_PATH, 'common_voice/cv_corpus_v1')

# Where to generate the .txt files, e.g. /home/user/../data/<target>.txt
TXT_TARGET_PATH = './data/'


def generate_list(dataset_path, dataset_name, target, dry_run=False):
    """Generate *.txt files containing the audio path and the corresponding sentence.
    Generated files are being stored at `TXT_TARGET_PATH`.

    Return additional data set information, see below.

    Args:
        dataset_path (str):
            Path the data set. Must match the `loader`'s capabilities.
        dataset_name (str):
            Name of the dataset. Supported dataset's:
            'timit', 'libri_speech', 'tedlium'
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
        'timit': _timit_loader,
        'libri_speech': _libri_speech_loader,
        'tedlium': _tedlium_loader,
        'common_voice': _common_voice_loader
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

    # RegEX filter pattern for valid characters.
    pattern = re.compile(r'[^a-z ]+')

    # Load the output string. Format ['/path/s.wav label text\n', ...]
    output = loader(dataset_path, target, pattern)

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


def _common_voice_loader(dataset_path, target, pattern):
    """Build the output string that can be written to the desired *.txt file.

    Uses only the valid datasets, additional constraints are:
    * Downvotes must be at maximum 1/5 of upvotes.
    * Valid accents are: 'us', 'england', 'canada', 'australia'.
    * Accepting samples with only 1 upvote at the moment.

    Args:
        dataset_path (str): Path of the data set.
        target (str): 'train', 'test', or 'dev'
        pattern (str): RegEx pattern that is used as whitelist for the written label texts.

    Returns:
        [str]: List containing the output string that can be written to *.txt file.
    """
    # Folders for each target.
    train_folders = ['cv-valid-train']
    test_folders = ['cv-valid-test']
    dev_folders = ['cv-valid-dev']

    # Assign target folders.
    if target == 'train':
        folders = train_folders
    elif target == 'test':
        folders = test_folders
    else:
        folders = dev_folders

    # Define valid accents. Review if '' should be accepted as well.
    valid_accents = ['us', 'england', 'canada', 'australia']

    output = []
    for folder in folders:
        # Open .csv file.
        with open('{}.csv'.format(os.path.join(dataset_path, folder))) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_lines = list(csv_reader)
            # print('csv_header:', csv_lines[0])

            for line in csv_lines[1:]:
                # Cleanup label text.
                text = re.sub(pattern, '', line[1]).strip().replace('  ', ' ')
                # Enfore min label length.
                if len(text) > 1:
                    # Review: Accept only 2 upvote examples, like documented?
                    # Check upvotes vs downvotes.
                    if int(line[2]) >= 1 and int(line[3]) / int(line[2]) <= 1/5:
                        # Check if speaker accent is valid.
                        if line[6] in valid_accents:
                            mp3_path = os.path.join(dataset_path, line[0])
                            assert os.path.isfile(mp3_path)
                            wav_path = '{}.wav'.format(mp3_path[:-4])

                            storage.delete_file_if_exists(wav_path)
                            # Convert .mp3 file into .wav file, reduce volume to 0.95,
                            # downsample to 16kHz and mono sound.
                            subprocess.call(['sox', '-v', '0.95', mp3_path, '-r', '16k',
                                             wav_path, 'remix', '1'])
                            assert os.path.isfile(wav_path)

                            # Add dataset relative to dataset path, label to .txt file buffer.
                            wav_path = os.path.relpath(wav_path, DATASET_PATH)
                            output.append('{} {}\n'.format(wav_path, text))

    return output


def _libri_speech_loader(dataset_path, target, pattern):
    """Build the output string that can be written to the desired *.txt file.

    Args:
        dataset_path (str): Path of the data set.
        target (str): 'train', 'test', or 'dev'
        pattern (str): RegEx pattern that is used as whitelist for the written label texts.

    Returns:
        [str]: List containing the output string that can be written to *.txt file.
    """
    # Folders for each target.
    train_folders = ['train-clean-100', 'train-clean-360']
    test_folders = ['test-clean']
    dev_folders = ['dev-clean']

    # Assign target folders.
    if target == 'train':
        folders = train_folders
    elif target == 'test':
        folders = test_folders
    else:
        folders = dev_folders

    output = []
    for folder in [os.path.join(dataset_path, f) for f in folders]:
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
                    # Absolute path.
                    wav_path = os.path.join(root, '{}.wav'.format(file_id))
                    assert os.path.isfile(wav_path), '{} not found.'.format(wav_path)

                    # Relative path to `DATASET_PATH`.
                    wav_path = os.path.relpath(wav_path, DATASET_PATH)

                    txt = re.sub(pattern, '', txt).strip().replace('  ', ' ')
                    output.append('{} {}\n'.format(wav_path, txt.strip()))

    return output


def _tedlium_loader(dataset_path, target, pattern):
    """Build the output string that can be written to the desired *.txt file.

     Note:
         Since TEDLIUM data is one large .wav file per speaker. Therefore this method creates
         several smaller partial .wav files. This takes some time.

    Note:
        The large .wav files are being converted into parts, even if `dry_run=True` has been
        selected.

    Args:
        dataset_path (str): Path of the data set.
        target (str): 'train', 'test', or 'dev'
        pattern (str): RegEx pattern that is used as whitelist for the written label texts.

    Returns:
        [str]: List containing the output string that can be written to *.txt file.
    """

    def seconds_to_sample(seconds, sr=16000):
        return int(seconds * sr)

    def write_wav_part(data, path, start, end, sr=16000):
        assert 0. <= start < (len(wav_data) / sr)
        assert start < end <= (len(wav_data) / sr)

        # print('Saving {:12,d}({:6.2f}s) to {:12,d}({:6.2f}s) at: {}'
        #       .format(seconds_to_sample(start), start, seconds_to_sample(end), end, path))

        storage.delete_file_if_exists(path)
        wavfile.write(path, sr, data[seconds_to_sample(start): seconds_to_sample(end)])

    target_folders = {
        'dev': 'dev',
        'test': 'test',
        'train': 'train'
    }

    target_folder = os.path.join(dataset_path, 'TEDLIUM_release2', target_folders[target], 'stm')

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

            wav_path = os.path.join(dataset_path, 'TEDLIUM_release2', target_folders[target],
                                    'sph', '{}.wav'.format(os.path.splitext(stm_file)[0]))
            assert os.path.isfile(wav_path), '{} not found.'.format(wav_path)

            # Load the audio data, to later split it into a part per audio segment.
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

                # Relative path to DATASET_PATH.
                part_path = os.path.relpath(part_path, DATASET_PATH)

                # Sanitize lines.
                text = text.lower()
                # Remove ` '`. TEDLIUM transcribes `i'm` as `i 'm`.
                text = text.replace(" '", '')
                text = re.sub(pattern, '', text).replace('  ', ' ').strip()
                # Remove labels with less than 5 words.
                if len(text.split(' ')) < 5:
                    continue
                output.append('{} {}\n'.format(part_path, text))

    return output


def _timit_loader(dataset_path, target, pattern):
    """Build the output string that can be written to the desired *.txt file.

    Args:
        dataset_path (str): Path of the data set.
        target (str): Not used at the moment for TIMIT data set. Set to `None`.
        pattern (str): RegEx pattern that is used as whitelist for the written label texts.

    Returns:
        [str]: List containing the output string that can be written to *.txt file.
    """
    if target != 'test' and target != 'train':
        raise ValueError('Timit only supports \'train\' and \'test\' targets.')

    # Location of timit intern .txt file listings.
    train_txt_path = os.path.join(dataset_path, 'train_all.txt')
    test_txt_path = os.path.join(dataset_path, 'test_all.txt')

    # Select target.
    master_txt_path = train_txt_path if target == 'train' else test_txt_path
    if not os.path.isfile(master_txt_path):
        raise ValueError('"{}" is not a file.'.format(master_txt_path))

    with open(master_txt_path, 'r') as f:
        master_data = f.readlines()

    output = []

    for line in master_data:
        wav_path, txt_path, _, _ = line.split(',')
        txt_path = os.path.join(dataset_path, txt_path)

        # Skip SAx.WAV files, since they are repeated by every speaker in the dataset.
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

            # Absolute path.
            wav_path = os.path.join(dataset_path, wav_path)

            # Relative path to `DATASET_PATH`.
            wav_path = os.path.relpath(wav_path, DATASET_PATH)

            output.append('{} {}\n'.format(wav_path, txt))

    return output


if __name__ == '__main__':
    __dry_run = False

    # TEDLIUMv2
    # generate_list(TEDLIUM_PATH, 'tedlium', 'test', dry_run=__dry_run)
    # generate_list(TEDLIUM_PATH, 'tedlium', 'dev', dry_run=__dry_run)
    # generate_list(TEDLIUM_PATH, 'tedlium', 'train', dry_run=__dry_run)

    # TIMIT
    # generate_list(TIMIT_PATH, 'timit', 'test', dry_run=__dry_run)
    # generate_list(TIMIT_PATH, 'timit', 'train', dry_run=__dry_run)

    # LibriSpeech ASR Corpus
    # generate_list(LIBRI_SPEECH_PATH, 'libri_speech', 'test', dry_run=__dry_run)
    # generate_list(LIBRI_SPEECH_PATH, 'libri_speech', 'dev', dry_run=__dry_run)
    # generate_list(LIBRI_SPEECH_PATH, 'libri_speech', 'train', dry_run=__dry_run)

    # Mozilla Common Voice
    generate_list(COMMON_VOICE_PATH, 'common_voice', 'test', dry_run=__dry_run)
    generate_list(COMMON_VOICE_PATH, 'common_voice', 'dev', dry_run=__dry_run)
    generate_list(COMMON_VOICE_PATH, 'common_voice', 'train', dry_run=__dry_run)
