"""Generate `train.txt` and `test.txt` for the `TIMIT`_ data set.
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


DATA_PATH = '/home/marc/workspace/speech/data/timit/TIMIT/'     # Path to the TIMIT data set.
TARGET_PATH = '/home/marc/workspace/speech/data/'               # Where to generate the .txt files.


def _gen_list(target):
    """Generate .txt files containing the audio path and the corresponding sentence.
    Return additional data set information, see below.

    Args:
        target (str): 'train' or 'test'

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

    for line in master_data:
        wav_path, txt_path, _, _ = line.split(',')
        txt_path = os.path.join(DATA_PATH, txt_path)

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

        output_line = '{} {}\n'.format(wav_path, txt)
        result.append(output_line)

    # Remove new line from last entry.
    result[-1] = result[-1].strip()

    target_path = os.path.join(TARGET_PATH, '{}.txt'.format(target))
    _delete_file_if_exists(target_path)

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
    train_char_s, train_word_s, train_len = _gen_list('train')
    test_char_s, test_word_s, test_len = _gen_list('test')
    print('#(TEST_WORD\\TRAIN_WORD)={}:'
          .format(len(test_word_s - train_word_s)), test_word_s - train_word_s)

    print('Training (train.txt : {}) and evaluation (test.txt : {}) file lists created.'
          .format(train_len, test_len))
