"""Load the LibriSpeech ASR corpus."""

import os
import sys
import subprocess

from tqdm import tqdm
from scipy.io import wavfile

from python.params import MIN_EXAMPLE_LENGTH, MAX_EXAMPLE_LENGTH
from python.dataset.config import CACHE_DIR, CORPUS_DIR
from python.dataset import download
from python.dataset.txt_files import generate_txt


# L8ER Add the `other` datasets as well and see if they improve the results.
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
    """Download, extract and build the output strings that can be written to the desired TXT files.

    L8ER: Can this be parallelized?

    Args:
        keep_archive (bool): Keep or delete the downloaded archive afterwards.

    Returns:
        Tuple[str]: Tuple containing the output string that can be written to TXT files.
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

    txt_paths = []
    for target in targets:
        # Generate the WAV and a string for the `<target>.txt` file.
        output = __libri_speech_loader(target['folders'])
        # Generate the `<target>.txt` file.
        txt_paths.append(generate_txt(__NAME, target['name'], output))

    # Cleanup extracted folder.
    download.cleanup_cache(__FOLDER_NAME)

    return tuple(txt_paths)


def __libri_speech_loader(folders):
    """Build the output string that can be written to the desired *.txt file.

    Args:
        folders (List[str]): List of directories to include, e.g.
            `['train-clean-100', 'train-clean-360']`

    Returns:
        [str]: List containing the output string that can be written to *.txt file.
    """
    if not os.path.isdir(__SOURCE_PATH):
        raise ValueError('"{}" is not a directory.'.format(__SOURCE_PATH))

    output = []
    folders_ = [os.path.join(__SOURCE_PATH, f) for f in folders]
    for folder in tqdm(folders_, desc='Converting Libri Speech data', total=len(folders_),
                       file=sys.stdout, dynamic_ncols=True, unit='Folder'):
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
                    flac_path = os.path.join(root, '{}.flac'.format(file_id))
                    assert os.path.isfile(flac_path), '{} not found.'.format(flac_path)

                    # Convert FLAC file WAV file and move it to the `data/corpus/..` directory.
                    wav_path = os.path.join(root, '{}.wav'.format(file_id))
                    wav_path = os.path.join(CORPUS_DIR, os.path.relpath(wav_path, CACHE_DIR))
                    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
                    subprocess.call(['sox', '-v', '0.95', flac_path, '-r', '16k', wav_path,
                                     'remix', '1'])
                    assert os.path.isfile(wav_path), '{} not found.'.format(wav_path)

                    # Validate that the example length is within boundaries.
                    (sr, y) = wavfile.read(wav_path)
                    length_sec = len(y) / sr
                    if not MIN_EXAMPLE_LENGTH <= length_sec <= MAX_EXAMPLE_LENGTH:
                        continue

                    # Relative path to `DATASET_PATH`.
                    wav_path = os.path.relpath(wav_path, CORPUS_DIR)

                    output.append('{} {}\n'.format(wav_path, txt.strip()))

    return output


# Test download script.
if __name__ == '__main__':
    print('Libri Speech txt_paths: ', libri_speech_loader(True))
    print('\nDone.')
