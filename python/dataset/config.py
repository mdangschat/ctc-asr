"""Configuration file for dataset creation. Also reference `python/dataset/generate_dataset.py`."""

import os

from python.params import BASE_PATH


CACHE_DIR = os.path.join(BASE_PATH, 'data/cache')
CORPUS_DIR = os.path.join(BASE_PATH, 'data/corpus')

# Where to generate the .txt files, e.g. /home/user/../<project_name>/data/<target>.txt
TXT_DIR = os.path.join(BASE_PATH, 'data')
