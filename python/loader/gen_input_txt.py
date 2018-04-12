"""Generate `all_train.txt` and `all_test.txt` for the `TIMIT`_ data set.
Additionally some information about the data set is being printed out.

Generated data format:
    `path/to/sample.wav transcription of the sample wave file<new_line>`

    The transcription is in lower case letters a-z with every word separated
    by a <space>. Punctuation is removed.

.. _TIMIT:
    https://vcs.zwuenf.org/agct_data/timit
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

from loader import audio_set_info


DATA_PATH = '/home/marc/workspace/speech/data/timit/TIMIT/'     # Path to the TIMIT data set.
TARGET_PATH = '/home/marc/workspace/speech/data/'               # Where to generate the .txt files.


def _gen_list(target, additional_output=False, dry_run=False):
    """Generate .txt files containing the audio path and the corresponding sentence.
    Return additional data set information, see below.
    review Documentation

    Args:
        target (str): 'train' or 'test'
        additional_output (bool): TODO Document & Implement

    Returns:
        char_set: Set containing each character within the data set.
        word_set: Set containing each word within the data set.
        num_samples (int): The number of samples in the data set.
    """

    if target != 'test' and target != 'train':
        raise ValueError('"{}" is not a valid target.'.format(target))

    master_path = os.path.join(DATA_PATH, '{}_all.txt'.format(target))

    if not os.path.isfile(master_path):
        raise ValueError('"{}" is not a file.'.format(master_path))

    with open(master_path, 'r') as f:
        master_data = f.readlines()

    result = []
    word_set = set()
    char_set = set()
    longest = 0
    shortest = (2 ** 31) - 1
    pattern = re.compile(r'[^a-zA-Z ]+')

    sample_len_list = []
    for i, line in enumerate(master_data):
        wav_path, txt_path, _, _ = line.split(',')
        txt_path = os.path.join(DATA_PATH, txt_path)

        basename = os.path.basename(wav_path)
        if 'SA1.WAV' == basename or 'SA2.WAV' == basename:
            continue

        with open(txt_path, 'r') as f:
            txt = f.readlines()
            assert len(txt) == 1, 'Text file contains to many lines. ({})'.format(txt_path)
            txt = txt[0].split(' ', 2)[2]

            txt = re.sub(pattern, '', txt).strip()
            txt = txt.replace('  ', ' ')
            txt = txt.lower()
            longest = len(txt) if len(txt) > longest else longest
            shortest = len(txt) if len(txt) < shortest else shortest
            char_set.update(set(list(txt)))
            word_set.update(set(txt.split(' ')))

        if additional_output:
            sample_len, _, _ = audio_set_info.sample_info(os.path.join(DATA_PATH, wav_path))
            sample_len_list.append(sample_len)

        output_line = '{} {}\n'.format(wav_path, txt)
        result.append(output_line)

    # Remove new line from last entry.
    result[-1] = result[-1].strip()

    target_path = os.path.join(TARGET_PATH, '{}.txt'.format(target))
    _delete_file_if_exists(target_path)

    if not dry_run:
        with open(target_path, 'w') as f:
            print('Writing {} lines of {} files to {}'.format(len(result), target, target_path))
            f.writelines(result)

    # Remove unwanted elements from set.
    char_set.discard('')
    char_set.discard(' ')
    word_set.discard('')
    word_set.discard(' ')

    # Print some information about the labels.
    print('#char_set={}:'.format(len(char_set)), char_set)
    print('#word_set={}:'.format(len(word_set)), word_set)
    print('Longest sentence was {} and the shortest was {} characters.'.format(longest, shortest))

    if additional_output:
        # MFCC sample information's.
        sample_len_list = np.array(sample_len_list, dtype=np.float32)
        sample_len_list = np.sort(sample_len_list)
        avg_value = np.mean(sample_len_list)
        min_value = np.amin(sample_len_list)
        max_value = np.amax(sample_len_list)
        plt.hist(sample_len_list, 64)
        plt.grid()
        plt.show()
        print('Sample lengths: min={}, max={}, avg={}'.format(min_value, max_value, avg_value))

        # Bin information.
        list_len = len(sample_len_list)
        num_bins = 16
        step = list_len // num_bins

        output = ''
        for i in range(0, list_len, step):
            output += ', {}'.format(int(sample_len_list[i]))
        print('Bins: ', output[2:])

    return char_set, word_set, len(result)


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
    print('Starting...')
    train_char_s, train_word_s, train_len = _gen_list('train', additional_output=True)
    test_char_s, test_word_s, test_len = _gen_list('test', additional_output=False)
    print('#(TEST_WORD\\TRAIN_WORD)={}:'
          .format(len(test_word_s - train_word_s)), test_word_s - train_word_s)

    print('Training (train.txt : {}) and evaluation (test.txt : {}) file lists created.'
          .format(train_len, test_len))
