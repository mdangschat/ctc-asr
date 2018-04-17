"""Generate `train.txt` and `test.txt` for the `LibriSpeech`_ data set.
 Additionally some information about the data set can be printed out.

Generated data format:
    `path/to/sample.wav transcription of the sample wave file<new_line>`

    The transcription is in lower case letters a-z with every word separated
    by a <space>. Punctuation is removed.

.. _LibriSpeech:
    http://openslr.org/12
"""

import os


# Path to the LibriSpeech ASR data set.
DATA_PATH = '/home/marc/workspace/speech/data/libri_speech/LibriSpeech/'
# Where to generate the .txt files.
TARGET_PATH = '/home/marc/workspace/speech/data/'


def _gen_list(target, additional_output=False, dry_run=False):
    """Generate .txt files containing the audio path and the corresponding sentence.
    Return additional data set information, see below.
    # TODO: Update documentation

    Args:
        target (str):
            'train' or 'test'
        additional_output (bool):
            Convert the audio data to features and extract additional information.
            Prints out optimal bucket sizes.
        dry_run (bool):
            Dry run, do not create output.txt file.

    Returns:
        Nothing.
    """

    if target != 'test' and target != 'train':
        raise ValueError('"{}" is not a valid target.'.format(target))

    if not os.path.isdir(DATA_PATH):
        raise ValueError('"{}" is not a directory.'.format(DATA_PATH))

    output = []
    for root, dirs, files in os.walk(DATA_PATH):
        if len(dirs) is 0:
            # Get list of `.trans.txt` files.
            trans_txt_files = [f for f in files if f.endswith('.trans.txt')]
            # Verify that a `*.trans.txt` file exists.
            assert len(trans_txt_files) is 1

            # Absolute path.
            trans_txt_path = os.path.join(root, trans_txt_files[0])

            # Load `.trans.txt` contents.
            with open(trans_txt_path, 'r') as f:
                lines = f.readlines()

            # Sanitize lines.
            lines = [line.lower().strip().split(' ', 1) for line in lines]

            for _id, txt in lines:
                path = os.path.join(root, '{}.flac'.format(_id))
                assert os.path.isfile(path)
                output.append('{} {}\n'.format(path, txt.strip()))

    if not dry_run:
        target_path = os.path.join(TARGET_PATH, '{}.txt'.format(target))
        _delete_file_if_exists(target_path)

        with open(target_path, 'w') as f:
            print('Writing {} lines of {} files to {}'.format(len(output), target, target_path))
            f.writelines(output)


def _delete_file_if_exists(path):
    """Delete the file for the given path, if it exists.

    Args:
        path (str): File path.

    Returns:
        Nothing.
    """
    if os.path.exists(path) and os.path.isfile(path):
        os.remove(path)


if __name__ == '__main__':
    print('Starting to generate .txt files...')
    _gen_list('train', additional_output=False)

    # TODO: No test set yet.
    # _gen_list('test', additional_output=False)
