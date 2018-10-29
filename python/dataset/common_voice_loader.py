"""Load the Mozilla Common Voice dataset."""

import sys
import os
import csv
import subprocess

from multiprocessing import Pool, Lock, cpu_count
from tqdm import tqdm
from scipy.io import wavfile

from python.util.storage import delete_file_if_exists
from python.params import MIN_EXAMPLE_LENGTH, MAX_EXAMPLE_LENGTH, BASE_PATH


# Path to the Mozilla Common Voice dataset.
__COMMON_VOICE_URL = 'https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz'
__DATASETS_PATH = os.path.join(BASE_PATH, '../datasets/speech_data')
__COMMON_VOICE_PATH = os.path.realpath(os.path.join(__DATASETS_PATH, 'common_voice/cv_corpus_v1'))

# Define valid accents.
__VALID_ACCENTS = ['us', 'england', 'canada', 'australia', 'wales', 'newzealand', 'ireland',
                   'scotland', 'wales', '']


def common_voice_loader(target):
    """Build the output string that can be written to the desired *.txt file.

    Uses only the valid datasets, additional constraints are:
    * Downvotes must be at maximum 1/4 of upvotes.
    * Valid accents are: 'us', 'england', 'canada', 'australia'.
    * Accepting samples with only 1 upvote at the moment.

    Args:
        target (str): 'train', 'test', or 'dev'

    Returns:
        List[str]: List containing the output string that can be written to *.txt file.
    """
    if not os.path.isdir(__COMMON_VOICE_PATH):
        raise ValueError('"{}" is not a directory.'.format(__COMMON_VOICE_PATH))

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

    output = []
    for folder in tqdm(folders, desc='Converting Common Voice data', total=len(folders),
                       file=sys.stdout, unit='CSVs', dynamic_ncols=True):
        # Open .csv file.
        with open('{}.csv'.format(os.path.join(__COMMON_VOICE_PATH, folder)), 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_lines = list(csv_reader)
            # print('csv_header:', csv_lines[0])
            # filename,text,up_votes,down_votes,age,gender,accent,duration

            lock = Lock()
            with Pool(processes=cpu_count()) as pool:
                # First line contains header.
                for result in pool.imap_unordered(__common_voice_loader_helper,
                                                  csv_lines[1:], chunksize=1):
                    if result is not None:
                        lock.acquire()
                        output.append(result)
                        lock.release()

    return output


def __common_voice_loader_helper(line):
    # Cleanup label text.
    text = line[1].strip().replace('  ', ' ')
    # Enforce min label length.
    if len(text) > 1:
        # Review: Accept only 2 upvote examples, like documented?
        # Check upvotes vs downvotes.
        if int(line[2]) >= 1 and int(line[3]) / int(line[2]) <= 1 / 4:
            # Check if speaker accent is valid.
            if line[6] in __VALID_ACCENTS:
                mp3_path = os.path.join(__COMMON_VOICE_PATH, line[0])
                assert os.path.isfile(mp3_path)
                wav_path = '{}.wav'.format(mp3_path[:-4])

                delete_file_if_exists(wav_path)
                # Convert MP3 to WAV, reduce volume to 0.95, downsample to 16kHz and mono sound.
                subprocess.call(['sox', '-v', '0.95', mp3_path, '-r', '16k', wav_path,
                                 'remix', '1'])
                assert os.path.isfile(wav_path)

                # Validate that the example length is within boundaries.
                (sr, y) = wavfile.read(wav_path)
                length_sec = len(y) / sr
                if not MIN_EXAMPLE_LENGTH <= length_sec <= MAX_EXAMPLE_LENGTH:
                    return None

                # Add dataset relative to dataset path, label to TXT file buffer.
                wav_path = os.path.relpath(wav_path, __DATASETS_PATH)
                return '{} {}\n'.format(wav_path, text)

    return None
