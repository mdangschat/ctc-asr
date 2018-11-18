"""
Configuration file for dataset creation.
Also reference `python/dataset/generate_dataset.py`.
"""

import os
import re

from python.params import BASE_PATH, FLAGS

CACHE_DIR = os.path.join(BASE_PATH, 'data/cache')
CORPUS_DIR = os.path.join(BASE_PATH, 'data/corpus')

# Where to generate the .txt files, e.g. /home/user/../<project_name>/data/<target>.txt
CSV_DIR = os.path.join(BASE_PATH, 'data')

CSV_DELIMITER = ';'

# CSV field names. The field order is always the same as this list from top to bottom.
CSV_HEADER_PATH = 'path'
CSV_HEADER_LABEL = 'label'
CSV_HEADER_LENGTH = 'length'

# (Whitelist) RegEX filter pattern for valid characters.
LABEL_WHITELIST_PATTERN = re.compile(r'[^a-z ]+')


def sox_commandline(input_path, target_path):
    """
    Create the parametrized list of commands to convert some audio file into another format, using
    sox.
    See `man sox`.

    Args:
        input_path (str):
            Path to the audio file that should be converted. With file extension.

        target_path (str):
            Path to where the converted file should be stored. With file extension.
    Returns:
        List[str]: List containing the call parameters for `subprocess.call`.
    """

    return ['sox',
            '-V1',  # Verbosity set to only errors (default is 2).
            '--volume', '0.95',
            input_path,
            '--rate', str(FLAGS.sampling_rate),
            target_path,
            'remix', '1'  # Mono channel.
            ]
