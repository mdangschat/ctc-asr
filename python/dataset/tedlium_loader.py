"""
Load the `TEDLIUM`_ (v2) dataset.

.. _TEDLIUM:
    http://openslr.org/19
"""

import math
import os
import re
import subprocess
import sys
from multiprocessing import Pool, Lock, cpu_count

from scipy.io import wavfile
from tqdm import tqdm

from python.dataset import download
from python.dataset.config import CACHE_DIR, CORPUS_DIR, sox_commandline
from python.dataset.config import CSV_HEADER_PATH, CSV_HEADER_LABEL, CSV_HEADER_LENGTH
from python.dataset.csv_file_helper import generate_csv
from python.params import MIN_EXAMPLE_LENGTH, MAX_EXAMPLE_LENGTH, FLAGS
from python.util.storage import delete_file_if_exists

# Path to the Tedlium v2 dataset.
__URL = 'http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz'
__MD5 = '7ffb54fa30189df794dcc5445d013368'
__NAME = 'tedlium'
__FOLDER_NAME = 'TEDLIUM_release2'
__SOURCE_PATH = os.path.join(CACHE_DIR, __FOLDER_NAME)
__TARGET_PATH = os.path.realpath(os.path.join(CORPUS_DIR, __FOLDER_NAME))

# Flag that marks time segments that should be skipped.
__IGNORE_FLAG = 'ignore_time_segment_in_scoring'

# RegEx pattern to extract TEDLIUM's .stm information's.
__PATTERN = re.compile(
    r"[.\w]+ [0-9] [.\w]+ ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) <[\w,]+> ([\w ']+)")


def tedlium_loader(keep_archive):
    """
    Download, extract and convert the TEDLIUM archive.
    Then build all possible CSV files (e.g. `<dataset_name>_train.csv`, `<dataset_name>_test.csv`).

    Requires lots of disk space, since the original format (SPH) is converted to WAV and then split
    up into parts.

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
            'folder': 'train'
        }, {
            'name': 'test',
            'folder': 'test'
        }, {
            'name': 'dev',
            'folder': 'dev'
        }
    ]

    txt_paths = []
    for target in targets:
        # Create target folder if necessary.
        target_directory = os.path.join(__TARGET_PATH, target['folder'], 'sph')
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Generate the WAV and a string for the `<target>.txt` file.
        source_directory = os.path.join(__SOURCE_PATH, target['folder'])
        output = __tedlium_loader(source_directory)
        # Generate the `<target>.txt` file.
        txt_paths.append(generate_csv(__NAME, target['name'], output))

    # Cleanup extracted folder.
    download.cleanup_cache(__FOLDER_NAME)

    return tuple(txt_paths)


def __tedlium_loader(target_folder):
    """
    Build the data that can be written to the desired CSV file.

     Note:
         Since TEDLIUM data is one large .wav file per speaker. Therefore this method creates
         several smaller partial .wav files. This takes some time.

        The large .wav files are being converted into parts, even if `dry_run=True` has been
        selected.

    Args:
        target_folder (str): E.g. `'train'`, `'test'`, or `'dev'`.

    Returns:
        List[Dict]: List containing the CSV dictionaries that can be written to the CSV file.
    """

    files = os.listdir(os.path.join(target_folder, 'stm'))

    lock = Lock()
    output = []
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(__tedlium_loader_helper,
                                               zip(files, [target_folder] * len(files))),
                           desc='Reading TEDLIUM files', total=len(files), file=sys.stdout,
                           unit='files', dynamic_ncols=True):
            if result is not None:
                lock.acquire()
                output.extend(result)
                lock.release()

    return output


def __tedlium_loader_helper(args):
    stm_file, target_folder = args
    if os.path.splitext(stm_file)[1] != '.stm':
        # This check is required, since there are swap files, etc. in the TEDLIUM dataset.
        print('WARN: Invalid .stm file found:', stm_file)
        return None

    stm_file_path = os.path.join(target_folder, 'stm', stm_file)
    with open(stm_file_path, 'r') as f:
        lines = f.readlines()

        sph_path = os.path.join(__SOURCE_PATH, target_folder, 'sph', '{}.sph'
                                .format(os.path.splitext(stm_file)[0]))
        assert os.path.isfile(sph_path), '{} not found.'.format(sph_path)

        wav_path = os.path.join(__SOURCE_PATH, target_folder, 'sph',
                                '{}.wav'.format(os.path.splitext(stm_file)[0]))

        # Convert SPH to WAV.
        subprocess.call(sox_commandline(sph_path, wav_path))
        assert os.path.isfile(wav_path), '{} not found.'.format(wav_path)

        # Load the audio data, to later split it into a part per audio segment.
        (sampling_rate, wav_data) = wavfile.read(wav_path)
        assert sampling_rate == FLAGS.sampling_rate

        output = []

        for i, line in enumerate(lines):
            if __IGNORE_FLAG in line:
                continue

            res = re.search(__PATTERN, line)
            if res is None:
                raise RuntimeError('TEDLIUM loader error in file {}\nLine: {}'
                                   .format(stm_file_path, line))

            start_time = float(res.group(1))
            end_time = float(res.group(2))
            text = res.group(3)

            # Create new partial .wav file.
            part_path = '{}_{}.wav'.format(wav_path[: -4], i)
            part_path = os.path.relpath(part_path, CACHE_DIR)
            part_path = os.path.join(CORPUS_DIR, part_path)
            __write_part_to_wav(wav_data, part_path, start_time, end_time)

            # Validate that the example length is within boundaries.
            (sr, y) = wavfile.read(part_path)
            length_sec = len(y) / sr
            if not MIN_EXAMPLE_LENGTH <= length_sec <= MAX_EXAMPLE_LENGTH:
                continue

            # Relative path to __DATASETS_PATH.
            part_path = os.path.relpath(part_path, CORPUS_DIR)

            # Sanitize lines.
            text = text.lower().replace(" '", '').replace('  ', ' ').strip()

            # Skip labels with less than 5 words.
            if len(text.split(' ')) > 4:
                output.append({
                    CSV_HEADER_PATH: part_path,
                    CSV_HEADER_LABEL: text.strip(),
                    CSV_HEADER_LENGTH: length_sec
                })

        return output


def __write_part_to_wav(wav_data, path, start, end, sr=16000):
    assert 0. <= start < (len(wav_data) / sr)
    assert start < end <= (len(wav_data) / sr)

    # print('Saving {:12,d}({:6.2f}s) to {:12,d}({:6.2f}s) at: {}'
    #       .format(seconds_to_sample(start), start, seconds_to_sample(end), end, path))

    delete_file_if_exists(path)
    wavfile.write(path, sr, wav_data[__seconds_to_sample(start, True):
                                     __seconds_to_sample(end, False)])


def __seconds_to_sample(seconds, start=True, sr=16000):
    if start:
        return int(math.floor(seconds * sr))
    else:
        return int(math.ceil(seconds * sr))


# Test download script.
if __name__ == '__main__':
    print('TEDLIUM csv_paths: ', tedlium_loader(True))
    print('\nDone.')
