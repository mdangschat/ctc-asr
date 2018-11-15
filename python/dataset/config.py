"""
Configuration file for dataset creation.
Also reference `python/dataset/generate_dataset.py`.
"""

import os
import re

from python.params import BASE_PATH

CACHE_DIR = os.path.join(BASE_PATH, 'data/cache')
CORPUS_DIR = os.path.join(BASE_PATH, 'data/corpus')

# Where to generate the .txt files, e.g. /home/user/../<project_name>/data/<target>.txt
CSV_DIR = os.path.join(BASE_PATH, 'data')

CSV_DELIMITER = ';'

# CSV field names
CSV_HEADER_PATH = 'path'
CSV_HEADER_LABEL = 'label'

# (Whitelist) RegEX filter pattern for valid characters.
LABEL_WHITELIST_PATTERN = re.compile(r'[^a-z ]+')
