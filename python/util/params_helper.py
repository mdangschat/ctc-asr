"""
Support routines for `python/params.py`.
"""

import json
import os

# Path to git root.
BASE_PATH = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

# Path to corpus description.
CORPUS_JSON_PATH = os.path.join(BASE_PATH, 'data/corpus.json')

if os.path.exists(CORPUS_JSON_PATH):
    with open(CORPUS_JSON_PATH, 'r') as f:
        json_data = json.load(f)

        TRAIN_SIZE = json_data['train_size']
        TEST_SIZE = json_data['test_size']
        DEV_SIZE = json_data['dev_size']
        BOUNDARIES = json_data['boundaries']
else:
    print('WARN: No "corpus.json" file present.')
    TRAIN_SIZE = TEST_SIZE = DEV_SIZE = -1
    BOUNDARIES = []
