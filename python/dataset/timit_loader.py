"""Load the TIMIT dataset."""

import os

from scipy.io import wavfile

from python.params import MIN_EXAMPLE_LENGTH, MAX_EXAMPLE_LENGTH, BASE_PATH


# Path to the TIMIT dataset.
__DATASETS_PATH = os.path.join(BASE_PATH, '../datasets/speech_data')
__TIMIT_PATH = os.path.realpath(os.path.join(__DATASETS_PATH, 'timit/TIMIT'))


def timit_loader(target):
    """Build the output string that can be written to the desired *.txt file.

    Args:
        target (str): 'train' or 'test'.

    Returns:
        [str]: List containing the output string that can be written to *.txt file.
    """
    if not os.path.isdir(__TIMIT_PATH):
        raise ValueError('"{}" is not a directory.'.format(__TIMIT_PATH))

    if target != 'test' and target != 'train':
        raise ValueError('Timit only supports "train" and "test" targets.')

    # Location of timit intern .txt file listings.
    train_txt_path = os.path.join(__TIMIT_PATH, 'train_all.txt')
    test_txt_path = os.path.join(__TIMIT_PATH, 'test_all.txt')

    # Select target.
    master_txt_path = train_txt_path if target == 'train' else test_txt_path
    if not os.path.isfile(master_txt_path):
        raise ValueError('"{}" is not a file.'.format(master_txt_path))

    with open(master_txt_path, 'r') as f:
        master_data = f.readlines()

    output = []

    for line in master_data:
        wav_path, txt_path, _, _ = line.split(',')
        txt_path = os.path.join(__TIMIT_PATH, txt_path)

        # Skip SAx.WAV files, since they are repeated by every speaker in the dataset.
        basename = os.path.basename(wav_path)
        if 'SA1.WAV' == basename or 'SA2.WAV' == basename:
            continue

        with open(txt_path, 'r') as f:
            txt = f.readlines()
            assert len(txt) == 1, 'Text file contains to many lines. ({})'.format(txt_path)
            txt = txt[0].split(' ', 2)[2]

            # Absolute path.
            wav_path = os.path.join(__TIMIT_PATH, wav_path)

            # Validate that the example length is within boundaries.
            (sr, y) = wavfile.read(wav_path)
            length_sec = len(y) / sr
            if not MIN_EXAMPLE_LENGTH <= length_sec <= MAX_EXAMPLE_LENGTH:
                continue

            # Relative path to `DATASET_PATH`.
            wav_path = os.path.relpath(wav_path, __DATASETS_PATH)

            output.append('{} {}\n'.format(wav_path, txt.strip()))

    return output
