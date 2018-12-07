"""
Calculate word based statistics for the train.csv file.
"""

import csv
import os
from collections import Counter

import numpy as np

from asr.dataset.config import CSV_DIR, CSV_DELIMITER, CSV_FIELDNAMES, CSV_HEADER_LABEL


def _load_labels(path):
    with open(path, 'r', encoding='utf-8') as file_handle:
        reader = csv.DictReader(file_handle, delimiter=CSV_DELIMITER, fieldnames=CSV_FIELDNAMES)
        csv_entries = list(reader)[1: -1]
        return [entry[CSV_HEADER_LABEL] for entry in csv_entries]


def _plot_word_stats(labels):
    # labels = labels[: 10]
    # For CTC: total_characters, top_10_characters, bottom_5_characters

    # ################# Word based stats ######################
    print('Word based statistics:')
    sentence_length = np.array([len(sentence.split(' ')) for sentence in labels])
    # Merge all sentences into one list.
    word_list = ' '.join(labels).split(' ')
    total_words = len(word_list)
    word_count = Counter(word_list)

    words_only_1 = [item for item in word_count.items() if item[1] <= 1]
    words_only_2 = [item for item in word_count.items() if item[1] <= 2]
    words_only_5 = [item for item in word_count.items() if item[1] <= 5]
    words_only_10 = [item for item in word_count.items() if item[1] <= 10]
    common_words = word_count.most_common()

    print('\ttotal_words = {:,d}\n'
          '\tnumber_unique_words = {:,d}\n'
          '\tmean_sentence_length = {:.2f} words\n'
          '\tmin_sentence_length = {:,d} words\n'
          '\tmax_sentence_length = {:,d} words'
          .format(total_words,
                  len(word_count.keys()),
                  np.mean(sentence_length),
                  np.min(sentence_length),
                  np.max(sentence_length)))
    print('\tMost common words: ', common_words[: 10])
    print('\t{} words occurred only 1 time; {:,d} words occurred only 2 times; '
          '{:,d} words occurred only 5 times; {:,d} words occurred only 10 times.'
          .format(len(words_only_1), len(words_only_2), len(words_only_5), len(words_only_10)))

    # ############## Character based stats #####################
    print('\nCharacter based statistics:')
    character_count = Counter()
    characters_per_label = []
    for label in labels:
        character_count += Counter(label)
        characters_per_label.append(len(label))
    characters_per_label = np.array(characters_per_label)

    total_characters = sum(character_count.values())
    common_characters = character_count.most_common()

    print('\ttotal_characters = {:,d}\n'
          '\tmean_label_length = {:.2f} characters\n'
          '\tmin_label_length = {:,d} characters\n'
          '\tmax_label_length = {:,d} characters'
          .format(total_characters,
                  np.mean(characters_per_label),
                  np.min(characters_per_label),
                  np.max(characters_per_label)))
    print('\tMost common characters:', common_characters)
    print('\tMost common characters:', [c[0] for c in common_characters])


if __name__ == '__main__':
    __CSV_PATH = os.path.join(CSV_DIR, 'train.csv')
    print('Calculating statistics for {}'.format(__CSV_PATH))

    _plot_word_stats(_load_labels(__CSV_PATH))
