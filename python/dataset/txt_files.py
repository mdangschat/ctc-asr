"""Helper methods to generate the train.txt files."""

import re
import os

from python.util import storage
from python.dataset.config import TXT_DIR


# (Whitelist) RegEX filter pattern for valid characters.
__PATTERN = re.compile(r'[^a-z ]+')


def generate_txt(dataset_name, target, output):
    """Generate *.txt files containing the audio path and the corresponding sentence.
    Generated files are being stored at `TXT_TARGET_PATH`.

    Return additional data set information, see below.

    Args:
        dataset_name (str):
            Name of the dataset, e.g. 'libri_speech'.
        target (str):
            Target name, e.g. 'train', 'test', 'dev'
        output (str):
            String containing the content for the `<dataset_name>_<target>.txt` file.

    Returns:
        (str): Path to the created TXT file.
    """

    target_txt_path = os.path.join(TXT_DIR, '{}_{}.txt'.format(dataset_name, target))
    print('Starting to generate: {}'.format(os.path.basename(target_txt_path)))

    # Remove illegal characters from labels.
    output = _remove_illegal_characters(output)

    # Filter out labels that are only shorter than 2 characters.
    output = list(filter(lambda x: len((x.split(' ', 1)[-1]).strip()) >= 2, output))

    # Write list to .txt file.
    print('> Writing {} lines of {} files to {}'.format(len(output), target, target_txt_path))
    # Delete the old file if it exists.
    storage.delete_file_if_exists(target_txt_path)

    # Write data to the file.
    with open(target_txt_path, 'w') as f:
        f.writelines(output)

    return target_txt_path


def _remove_illegal_characters(lines):
    """Remove every not whitelisted character from a list of formatted lines.

    Args:
        lines (List[str]): List if lines in the format '/path/to/file.wav label text here'

    Returns:
        List[str]: Filtered list of lines.
    """
    result = []
    for line in lines:
        path, text = line.split(' ', 1)
        text = re.sub(__PATTERN, '', text.lower()).strip().replace('  ', ' ')
        result.append('{} {}\n'.format(path, text))
    return result
