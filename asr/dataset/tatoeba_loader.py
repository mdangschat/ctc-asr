"""
Load the `Tatoeba`_ dataset.

.. _Tatoeba:
    https://tatoeba.org/eng/downloads
"""

import csv
import os
import subprocess
import sys
import time
from multiprocessing import Pool, Lock, cpu_count

from scipy.io import wavfile
from tqdm import tqdm

from asr.dataset import download
from asr.dataset.config import CACHE_DIR, CORPUS_DIR, sox_commandline
from asr.dataset.config import CSV_HEADER_PATH, CSV_HEADER_LABEL, CSV_HEADER_LENGTH
from asr.dataset.csv_file_helper import generate_csv
from asr.params import MIN_EXAMPLE_LENGTH, MAX_EXAMPLE_LENGTH
from asr.util.storage import delete_file_if_exists

# Path to the Taboeba dataset.
__URL = 'https://downloads.tatoeba.org/audio/tatoeba_audio_eng.zip'
__MD5 = 'd76252fd704734fc3d8bf5b44e029809'
__NAME = 'tatoeba'
__FOLDER_NAME = 'tatoeba_audio_eng'
__SOURCE_PATH = os.path.join(CACHE_DIR, __FOLDER_NAME)
__TARGET_PATH = os.path.realpath(os.path.join(CORPUS_DIR, __FOLDER_NAME))


def tatoeba_loader(keep_archive):
    """
    Download, extract and convert the Tatoeba archive.
    Then build all possible CSV files (e.g. `<dataset_name>_train.csv`, `<dataset_name>_test.csv`).

    Args:
        keep_archive (bool): Keep or delete the downloaded archive afterwards.

    Returns:
        List[str]: List containing the created CSV file paths.
    """

    # Download and extract the dataset if necessary.
    download.maybe_download(__URL, md5=__MD5, cache_archive=keep_archive)
    if not os.path.isdir(__SOURCE_PATH):
        raise ValueError('"{}" is not a directory.'.format(__SOURCE_PATH))

    # Download user ratings CSV file.
    csv_path = os.path.join(__SOURCE_PATH, 'users_sentences.csv')
    download.download_with_progress('http://downloads.tatoeba.org/exports/users_sentences.csv',
                                    csv_path)
    assert os.path.exists(csv_path)

    target = 'train'
    # Generate the WAV and a string for the `<target>.txt` file.
    output = __tatoeba_loader(target)
    # Generate the `<target>.txt` file.
    csv_path = generate_csv(__NAME, target, output)

    # Cleanup extracted folder.
    download.cleanup_cache(__FOLDER_NAME)

    return csv_path


def __tatoeba_loader(target):
    """
    Build the data that can be written to the desired CSV file.

    Args:
        target (str): Only 'train' is supported for the Tatoeba dataset.

    Returns:
        List[Dict]: List containing the CSV dictionaries that can be written to the CSV file.
    """
    if not os.path.isdir(__SOURCE_PATH):
        raise ValueError('"{}" is not a directory.'.format(__SOURCE_PATH))

    if target != 'train':
        raise ValueError('Invalid target. Tatoeba only has a train dataset.')

    validated_samples = set()  # Set of all sample IDs that have been validated.
    # Parse dataset meta data information to filter out low ranked samples.
    with open(os.path.join(__SOURCE_PATH, 'users_sentences.csv'), 'r') as csv_handle:
        csv_reader = csv.reader(csv_handle, delimiter='\t')
        csv_lines = list(csv_reader)
        # print('csv_header: username\tsentence_id\trating\tdate_added\tdate_modified')

        for username, _id, rating, _, _ in csv_lines:
            rating = int(rating)
            if rating >= 1:
                path = os.path.join(__SOURCE_PATH, 'audio', username, _id)
                validated_samples.add(path)

    samples = []  # List of dictionaries of all files and labels and in the dataset.
    # Parse dataset meta data information to filter out low ranked samples.
    with open(os.path.join(__SOURCE_PATH, 'sentences_with_audio.csv'), 'r') as csv_handle:
        csv_reader = csv.reader(csv_handle, delimiter='\t')
        csv_lines = list(csv_reader)
        csv_lines = csv_lines[1:]  # Remove CSV header.
        # print('csv_header: sentence_id\tusername\ttext')

        for _id, username, text in tqdm(csv_lines,
                                        desc='Parsing Tatoeba CSV', total=len(csv_lines),
                                        file=sys.stdout, unit='entries', dynamic_ncols=True):
            path = os.path.join(__SOURCE_PATH, 'audio', username, _id)
            if path in validated_samples:
                samples.append({'path': path, 'text': text})

    # Create target folder structure.
    for sample in samples:
        source_dir_path = os.path.join(os.path.relpath(sample['path'], __SOURCE_PATH), '..')
        target_dir_path = os.path.realpath(os.path.join(__TARGET_PATH, source_dir_path))

        if not os.path.exists(target_dir_path):
            os.makedirs(target_dir_path)

    lock = Lock()
    buffer = []
    missing_mp3_counter = 0
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(__tatoeba_loader_helper, samples, chunksize=1),
                           desc='Converting Tatoeba MP3 to WAV', total=len(samples),
                           file=sys.stdout, unit='files', dynamic_ncols=True):
            lock.acquire()
            if result is None:
                missing_mp3_counter += 1
            else:
                buffer.append(result)
            lock.release()

    print('WARN: {} MP3 files listed in the CSV could not be found.'
          .format(missing_mp3_counter))

    return buffer


def __tatoeba_loader_helper(sample):
    path = sample['path']
    text = sample['text']
    mp3_path = '{}.mp3'.format(path)
    wav_path = '{}.wav'.format(path)
    wav_path = os.path.join(__TARGET_PATH, os.path.relpath(wav_path, __SOURCE_PATH))

    # Check if audio file MP3 exists.
    if not os.path.isfile(mp3_path):
        # print('WARN: Audio file missing: {}'.format(mp3_path))
        return None

    # Make sure the file isn't empty.
    try:
        if os.path.getsize(mp3_path) <= 4048:
            return None
    except OSError:
        return None

    # If a WAV file with the desired name already exist, delete it.
    delete_file_if_exists(wav_path)

    # Convert MP3 file into WAV file, reduce volume to 0.95, downsample to 16kHz mono sound.
    # Note that this call produces the WAV files in the `data/corpus` directory.
    ret = subprocess.call(sox_commandline(mp3_path, wav_path))
    if not os.path.isfile(wav_path):
        raise RuntimeError('Failed to create WAV file with error code={}: {}'.format(ret, wav_path))

    # Validate that the example length is within boundaries.
    length_sec = None
    for i in range(5):
        try:
            (sr, y) = wavfile.read(wav_path)
            length_sec = len(y) / sr
            if not MIN_EXAMPLE_LENGTH <= length_sec <= MAX_EXAMPLE_LENGTH:
                return None
            break
        except ValueError:
            print('WARN: Could not load ({}/5) wavfile: {}'.format(i, wav_path))
            if i == 4:
                raise
            time.sleep(1)

    wav_path = os.path.relpath(wav_path, CORPUS_DIR)

    return {
        CSV_HEADER_PATH: wav_path,
        CSV_HEADER_LABEL: text.strip(),
        CSV_HEADER_LENGTH: length_sec
    }


# Test download script.
if __name__ == '__main__':
    print('Tatoeba csv_paths: ', tatoeba_loader(True))
    print('\nDone.')
