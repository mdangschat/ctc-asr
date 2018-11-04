"""Support routines for `python/params.py`."""

import os
import json

# Path to git root.
BASE_PATH = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
print("DEBUG BASE_PATH=", BASE_PATH)

JSON = os.path.join(BASE_PATH, 'data/corpus.json')
print("JSON_PATH:", JSON)

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
