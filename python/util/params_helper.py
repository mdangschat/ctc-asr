"""
Support routines for `python/params.py`.
"""

import json
import os

# Path to git root.
BASE_PATH = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

# Path to corpus description.
JSON = os.path.join(BASE_PATH, 'data/corpus.json')

if os.path.exists(JSON):
    with open(JSON, 'r') as f:
        data = json.load(f)

        TRAIN_SIZE = data['train_size']
        TEST_SIZE = data['test_size']
        DEV_SIZE = data['dev_size']
        BOUNDARIES = data['boundaries']
else:
    print('WARN: No "corpus.json" present.')
    TRAIN_SIZE = TEST_SIZE = DEV_SIZE = -1
    BOUNDARIES = []
