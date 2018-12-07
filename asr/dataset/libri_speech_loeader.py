"""
Load the `LibriSpeech`_ ASR corpus.

.. _LibriSpeech:
    http://openslr.org/12
"""

import os
import subprocess
import sys
from multiprocessing import Pool, Lock, cpu_count

from scipy.io import wavfile
from tqdm import tqdm

from asr.dataset import download
from asr.dataset.config import CACHE_DIR, CORPUS_DIR, sox_commandline
from asr.dataset.config import CSV_HEADER_PATH, CSV_HEADER_LABEL, CSV_HEADER_LENGTH
from asr.dataset.csv_file_helper import generate_csv
from asr.params import MIN_EXAMPLE_LENGTH, MAX_EXAMPLE_LENGTH

# L8ER: Add the `other` datasets as well and see if they improve the results.
# Path to the LibriSpeech ASR dataset.
__URLs = [
    'http://www.openslr.org/resources/12/dev-clean.tar.gz',
    'http://www.openslr.org/resources/12/test-clean.tar.gz',
    'http://www.openslr.org/resources/12/train-clean-100.tar.gz',
    'http://www.openslr.org/resources/12/train-clean-360.tar.gz'
]
__MD5s = [
    '42e2234ba48799c1f50f24a7926300a1',
    '32fa31d27d2e1cad72775fee3f4849a9',
    '2a93770f6d5c6c964bc36631d331a522',
    'c0e676e450a7ff2f54aeade5171606fa'
]
__NAME = 'librispeech'
__FOLDER_NAME = 'LibriSpeech'
__SOURCE_PATH = os.path.join(CACHE_DIR, __FOLDER_NAME)
__TARGET_PATH = os.path.realpath(os.path.join(CORPUS_DIR, __FOLDER_NAME))


def libri_speech_loader(keep_archive):
    """
    Download, extract and convert the Libri Speech archive.
    Then build all possible CSV files (e.g. `<dataset_name>_train.csv`, `<dataset_name>_test.csv`).

    Args:
        keep_archive (bool): Keep or delete the downloaded archive afterwards.

    Returns:
        List[str]: List containing the created CSV file paths.
    """

    # Download and extract the dataset if necessary.
    download.maybe_download_batch(__URLs, md5s=__MD5s, cache_archives=keep_archive)
    if not os.path.isdir(__SOURCE_PATH):
        raise ValueError('"{}" is not a directory.'.format(__SOURCE_PATH))

    # Folders for each target.
    targets = [
        {
            'name': 'train',
            'folders': ['train-clean-100', 'train-clean-360']
        }, {
            'name': 'test',
            'folders': ['test-clean']
        }, {
            'name': 'dev',
            'folders': ['dev-clean']
        }
    ]

    csv_paths = []
    for target in targets:
        # Generate the WAV and a string for the `<target>.txt` file.
        output = __libri_speech_loader(target['folders'])
        # Generate the `<target>.txt` file.
        csv_paths.append(generate_csv(__NAME, target['name'], output))

    # Cleanup extracted folder.
    download.cleanup_cache(__FOLDER_NAME)

    return csv_paths


def __libri_speech_loader(folders):
    """
    Build the data that can be written to the desired CSV file.

    Args:
        folders (List[str]): List of directories to include, e.g.
            `['train-clean-100', 'train-clean-360']`

    Returns:
        List[Dict]: List containing the CSV dictionaries that can be written to the CSV file.
    """
    if not os.path.isdir(__SOURCE_PATH):
        raise ValueError('"{}" is not a directory.'.format(__SOURCE_PATH))

    # Absolute folder paths.
    folders = [os.path.join(__SOURCE_PATH, folder) for folder in folders]
    # List of `os.walk` results. Tuple[str, List[str], List[str]]
    root_dir_files = []
    for folder in folders:
        root_dir_files.extend(os.walk(folder))
    # Remove non-leaf folders.
    root_dir_files = list(filter(lambda x: len(x[1]) == 0, root_dir_files))

    lock = Lock()
    output = []
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(__libri_speech_loader_helper, root_dir_files),
                           desc='Converting Libri Speech data', total=len(root_dir_files),
                           file=sys.stdout, dynamic_ncols=True, unit='directories'):
            if result is not None:
                lock.acquire()
                output.extend(result)
                lock.release()

        return output


def __libri_speech_loader_helper(args):
    root = args[0]
    dirs = args[1]
    files = args[2]

    if not dirs:
        return None

    # Get list of `.trans.txt` files.
    trans_txt_files = [f for f in files if f.endswith('.trans.txt')]
    # Verify that a `*.trans.txt` file exists.
    assert len(trans_txt_files) == 1, 'No .tans.txt file found: {}'.format(trans_txt_files)

    # Absolute path.
    trans_txt_path = os.path.join(root, trans_txt_files[0])

    # Load `.trans.txt` contents.
    with open(trans_txt_path, 'r') as f:
        lines = f.readlines()

    # Sanitize lines.
    lines = [line.lower().strip().split(' ', 1) for line in lines]

    buffer = []
    for file_id, label in lines:
        # Absolute path.
        flac_path = os.path.join(root, '{}.flac'.format(file_id))
        assert os.path.isfile(flac_path), '{} not found.'.format(flac_path)

        # Convert FLAC file WAV file and move it to the `data/corpus/..` directory.
        wav_path = os.path.join(root, '{}.wav'.format(file_id))
        wav_path = os.path.join(CORPUS_DIR, os.path.relpath(wav_path, CACHE_DIR))
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        subprocess.call(sox_commandline(flac_path, wav_path))
        assert os.path.isfile(wav_path), '{} not found.'.format(wav_path)

        # Validate that the example length is within boundaries.
        (sr, y) = wavfile.read(wav_path)
        length_sec = len(y) / sr
        if not MIN_EXAMPLE_LENGTH <= length_sec <= MAX_EXAMPLE_LENGTH:
            continue

        # Relative path to `DATASET_PATH`.
        wav_path = os.path.relpath(wav_path, CORPUS_DIR)

        buffer.append({
            CSV_HEADER_PATH: wav_path,
            CSV_HEADER_LABEL: label.strip(),
            CSV_HEADER_LENGTH: length_sec
        })

    return buffer


# Test download script.
if __name__ == '__main__':
    print('Libri Speech csv_paths: ', libri_speech_loader(True))
    print('\nDone.')
