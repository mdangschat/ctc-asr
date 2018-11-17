"""
Load the Mozilla `Common Voice`_ (v1) dataset.

.. Common Voice:
    https://voice.mozilla.org/en
"""

import csv
import os
import subprocess
import sys
from multiprocessing import Pool, Lock, cpu_count

from scipy.io import wavfile
from tqdm import tqdm

from python.dataset import download
from python.dataset.config import CACHE_DIR, CORPUS_DIR
from python.dataset.config import CSV_HEADER_PATH, CSV_HEADER_LABEL
from python.dataset.csv_file_helper import generate_csv
from python.params import MIN_EXAMPLE_LENGTH, MAX_EXAMPLE_LENGTH
from python.util.storage import delete_file_if_exists

# Path to the Mozilla Common Voice dataset.
__URL = 'https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz'
__MD5 = 'f1007e78cf91ab76b7cd3f1e8f554110'
__NAME = 'commonvoice'
__FOLDER_NAME = 'cv_corpus_v1'
__SOURCE_PATH = os.path.join(CACHE_DIR, __FOLDER_NAME)
__TARGET_PATH = os.path.realpath(os.path.join(CORPUS_DIR, __FOLDER_NAME))

# Define valid accents.
__VALID_ACCENTS = ['us',
                   'england',
                   'canada',
                   'australia',
                   'wales',
                   'newzealand',
                   'ireland',
                   'scotland',
                   'wales',
                   '']


def common_voice_loader(keep_archive):
    """
    Download, extract and convert the Common Voice archive.
    Then build all possible CSV files (e.g. `<dataset_name>_train.csv`, `<dataset_name>_test.csv`).

    Uses only the valid datasets, additional constraints are:
    * Downvotes must be at maximum 1/4 of upvotes.
    * Valid accents are: 'us', 'england', 'canada', 'australia'.
    * Accepting samples with only 1 upvote at the moment.

    Args:
        keep_archive (bool): Keep or delete the downloaded archive afterwards.

    Returns:
        List[str]: List containing the created CSV file paths.
    """

    # Download and extract the dataset if necessary.
    download.maybe_download(__URL, md5=__MD5, cache_archive=keep_archive)
    if not os.path.isdir(__SOURCE_PATH):
        raise ValueError('"{}" is not a directory.'.format(__SOURCE_PATH))

    # Folders for each target.
    targets = [
        {
            'name': 'train',
            'folders': ['cv-valid-train']
        }, {
            'name': 'test',
            'folders': ['cv-valid-test']
        }, {
            'name': 'dev',
            'folders': ['cv-valid-dev']
        }
    ]

    csv_paths = []
    for target in targets:
        # Generate the path and label for the `<target>.csv` file.
        output = __common_voice_loader(target['folders'])
        # Generate the `<target>.csv` file.
        csv_paths.append(generate_csv(__NAME, target['name'], output))

    # Cleanup extracted folder.
    download.cleanup_cache(__FOLDER_NAME)

    return csv_paths


def __common_voice_loader(folders):
    """
    Build the data that can be written to the desired CSV file.

    Uses only the valid datasets, additional constraints are:
    * Downvotes must be at maximum 1/4 of upvotes.
    * Valid accents are: 'us', 'england', 'canada', 'australia'.
    * Accepting samples with only 1 upvote at the moment.

    Args:
        folders (List[str]): A list containing folder names, e.g. `['train-valid', 'train-other']`.

    Returns:
        List[Dict]: List containing the CSV dictionaries that can be written to the CSV file.
    """

    output = []
    for folder in tqdm(folders, desc='Converting Common Voice data', total=len(folders),
                       file=sys.stdout, unit='files', dynamic_ncols=True):
        # Open .csv file.
        with open('{}.csv'.format(os.path.join(__SOURCE_PATH, folder)), 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_lines = list(csv_reader)
            # print('csv_header:', csv_lines[0])
            # filename,text,up_votes,down_votes,age,gender,accent,duration

            lock = Lock()
            with Pool(processes=cpu_count()) as pool:
                # Create target folder if necessary.
                target_directory = os.path.join(__TARGET_PATH, folder)
                if not os.path.exists(target_directory):
                    os.makedirs(target_directory)

                # First line contains header.
                for result in pool.imap_unordered(__common_voice_loader_helper,
                                                  csv_lines[1:], chunksize=1):
                    if result is not None:
                        lock.acquire()
                        output.append(result)
                        lock.release()

    return output


def __common_voice_loader_helper(line):
    # Helper method for thread pool.

    # Cleanup label text.
    text = line[1].strip().replace('  ', ' ')
    # Enforce min label length.
    if len(text) > 1:
        # Check upvotes vs downvotes.
        if int(line[2]) >= 1 and int(line[3]) / int(line[2]) <= 1 / 4:
            # Check if speaker accent is valid.
            if line[6] in __VALID_ACCENTS:
                mp3_path = os.path.join(__SOURCE_PATH, line[0])
                assert os.path.isfile(mp3_path)
                wav_path = os.path.relpath('{}.wav'.format(mp3_path[:-4]), __SOURCE_PATH)
                wav_path = os.path.join(__TARGET_PATH, wav_path)

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

                # Add dataset relative to dataset path, label to CSV file buffer.
                wav_path = os.path.relpath(wav_path, CORPUS_DIR)

                return {CSV_HEADER_PATH: wav_path, CSV_HEADER_LABEL: text}

    return None


# Test download script.
if __name__ == '__main__':
    print('Common Voice csv_paths: ', common_voice_loader(True))
    print('\nDone.')
