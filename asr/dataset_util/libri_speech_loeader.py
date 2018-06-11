"""Load the LibriSpeech ASR corpus."""

import os

from asr.params import BASE_PATH


# Path to the LibriSpeech ASR dataset.
__DATASETS_PATH = os.path.join(BASE_PATH, '../datasets/speech_data')
__LIBRI_SPEECH_PATH = os.path.realpath(os.path.join(__DATASETS_PATH, 'libri_speech/LibriSpeech'))


def libri_speech_loader(target):
    """Build the output string that can be written to the desired *.txt file.

    Args:
        target (str): 'train', 'test', or 'dev'

    Returns:
        [str]: List containing the output string that can be written to *.txt file.
    """
    if not os.path.isdir(__LIBRI_SPEECH_PATH):
        raise ValueError('"{}" is not a directory.'.format(__LIBRI_SPEECH_PATH))

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
    for folder in [os.path.join(__LIBRI_SPEECH_PATH, f) for f in folders]:
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
                    wav_path = os.path.relpath(wav_path, __DATASETS_PATH)

                    output.append('{} {}\n'.format(wav_path, txt.strip()))

    return output
