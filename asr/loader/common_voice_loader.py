"""Load the Mozilla Common Voice dataset."""

# TODO Incomplete

import sys
import os
import csv
import subprocess

from multiprocessing import Pool, Lock, cpu_count
from tqdm import tqdm

from asr.params import BASE_PATH
from asr.util.storage import delete_file_if_exists


# Path to the Mozilla Common Voice dataset.
__DATASETS_PATH = os.path.join(BASE_PATH, '../datasets/speech_data')
__COMMON_VOICE_PATH = os.path.realpath(os.path.join(__DATASETS_PATH, 'common_voice/cv_corpus_v1'))
print('__COMMON_VOICE_PATH:', __COMMON_VOICE_PATH)   # TODO


def common_voice_loader(target):
    """Build the output string that can be written to the desired *.txt file.

    Uses only the valid datasets, additional constraints are:
    * Downvotes must be at maximum 1/5 of upvotes.
    * Valid accents are: 'us', 'england', 'canada', 'australia'.
    * Accepting samples with only 1 upvote at the moment.

    Args:
        target (str): 'train', 'test', or 'dev'

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
        with open('{}.csv'.format(os.path.join(__COMMON_VOICE_PATH, folder)), 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_lines = list(csv_reader)
            # print('csv_header:', csv_lines[0])
            # filename,text,up_votes,down_votes,age,gender,accent,duration

            for line in csv_lines[1:]:
                # Cleanup label text.
                text = line[1].strip().replace('  ', ' ')
                # Enfore min label length.
                if len(text) > 1:
                    # Review: Accept only 2 upvote examples, like documented?
                    # Check upvotes vs downvotes.
                    if int(line[2]) >= 1 and int(line[3]) / int(line[2]) <= 1/5:
                        # Check if speaker accent is valid.
                        if line[6] in valid_accents:
                            mp3_path = os.path.join(__COMMON_VOICE_PATH, line[0])
                            assert os.path.isfile(mp3_path)
                            wav_path = '{}.wav'.format(mp3_path[:-4])

                            delete_file_if_exists(wav_path)
                            # Convert .mp3 file into .wav file, reduce volume to 0.95,
                            # downsample to 16kHz and mono sound.
                            subprocess.call(['sox', '-v', '0.95', mp3_path, '-r', '16k',
                                             wav_path, 'remix', '1'])
                            assert os.path.isfile(wav_path)

                            # Add dataset relative to dataset path, label to .txt file buffer.
                            wav_path = os.path.relpath(wav_path, __COMMON_VOICE_PATH)
                            output.append('{} {}\n'.format(wav_path, text))

    return output
