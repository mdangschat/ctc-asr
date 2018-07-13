"""Load the Tatoeba dataset."""

import sys
import os
import csv
import subprocess
import time

from multiprocessing import Pool, Lock, cpu_count
from tqdm import tqdm
from scipy.io import wavfile

from asr.params import BASE_PATH
from asr.util.storage import delete_file_if_exists
from asr.params import MIN_EXAMPLE_LENGTH, MAX_EXAMPLE_LENGTH


# Path to the Taboeba dataset.
__DATASETS_PATH = os.path.join(BASE_PATH, '../datasets/speech_data')
__TATOEBA_PATH = os.path.realpath(os.path.join(__DATASETS_PATH, 'tatoeba/tatoeba_audio_eng'))


def tatoeba_loader(target):
    """Build the output string that can be written to the desired *.txt file.

    Args:
        target (str): 'train'

    Returns:
        List[str]: List containing the output string that can be written to *.txt file.
    """
    if not os.path.isdir(__TATOEBA_PATH):
        raise ValueError('"{}" is not a directory.'.format(__TATOEBA_PATH))

    if target != 'train':
        raise ValueError('Invalid target. Tatoeba only has a train dataset.')

    validated_samples = set()     # Set of all sample IDs that have been validated.
    # Parse dataset meta data information to filter out low ranked samples.
    with open(os.path.join(__TATOEBA_PATH, 'users_sentences.csv'), 'r') as csv_handle:
        csv_reader = csv.reader(csv_handle, delimiter='\t')
        csv_lines = list(csv_reader)
        # print('csv_header: username\tsentence_id\trating\tdate_added\tdate_modified')

        for username, _id, rating, _, _ in csv_lines:
            rating = int(rating)
            if rating >= 1:
                path = os.path.join(__TATOEBA_PATH, 'audio', username, _id)
                validated_samples.add(path)

    samples = []     # List of dictionaries of all files and labels and in the dataset.
    # Parse dataset meta data information to filter out low ranked samples.
    with open(os.path.join(__TATOEBA_PATH, 'sentences_with_audio.csv'), 'r') as csv_handle:
        csv_reader = csv.reader(csv_handle, delimiter='\t')
        csv_lines = list(csv_reader)
        csv_lines = csv_lines[1:]  # Remove CSV header.
        # print('csv_header: sentence_id\tusername\ttext')

        for _id, username, text in tqdm(csv_lines,
                                        desc='Loading Tatoeba CSV', total=len(csv_lines),
                                        file=sys.stdout, unit='entries', dynamic_ncols=True):
            path = os.path.join(__TATOEBA_PATH, 'audio', username, _id)
            if path in validated_samples:
                samples.append({'path': path, 'text': text})

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

    # Check if audio file MP3 exists.
    if not os.path.isfile(mp3_path):
        # print('WARN: Audio file missing: {}'.format(mp3_path))
        return None

    # Check if file isn't empty.
    try:
        if os.path.getsize(mp3_path) <= 4048:
            return None
    except OSError:
        return None

    delete_file_if_exists(wav_path)

    # Convert MP3 file into WAV file, reduce volume to 0.95, downsample to 16kHz mono sound.
    ret = subprocess.call(['sox', '-v', '0.95', mp3_path, '-r', '16k', wav_path, 'remix', '1'])
    if not os.path.isfile(wav_path):
        raise RuntimeError('Failed to create WAV file with error code={}: {}'.format(ret, wav_path))

    # Validate that the example length is within boundaries.
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

    wav_path = os.path.relpath(wav_path, __DATASETS_PATH)
    return '{} {}\n'.format(wav_path, text.strip())
