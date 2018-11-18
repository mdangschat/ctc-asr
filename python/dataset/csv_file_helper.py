"""
Helper methods to generate the CSV files.
"""

import csv
import os
import re

from python.dataset.config import CSV_HEADER_LABEL, CSV_HEADER_LENGTH, CSV_FIELDNAMES
from python.dataset.config import LABEL_WHITELIST_PATTERN, CSV_DIR, CSV_DELIMITER
from python.input_functions import WIN_STEP
from python.util import storage
from python.util.matplotlib_helper import pyplot_display


def generate_csv(dataset_name, target, csv_data):
    """
    Generate CSV files containing the audio path and the corresponding sentence.
    Generated files are being stored at `CSV_TARGET_PATH`.
    Examples with labels consisting of one or two characters are omitted.

    Return additional data set information, see below.

    Args:
        dataset_name (str):
            Name of the dataset, e.g. 'libri_speech'.

        target (str):
            Target name, e.g. 'train', 'test', 'dev'

        csv_data (List[Dict]):
            List containing the csv content for the `<dataset_name>_<target>.csv` file.

    Returns:
        str: Path to the created CSV file.
    """

    target_csv_path = os.path.join(CSV_DIR, '{}_{}.csv'.format(dataset_name, target))
    print('Starting to generate: {}'.format(os.path.basename(target_csv_path)))

    # Remove illegal characters from labels.
    for csv_entry in csv_data:
        # Apply label whitelist filter.
        csv_entry[CSV_HEADER_LABEL] = re.sub(LABEL_WHITELIST_PATTERN,
                                             '',
                                             csv_entry[CSV_HEADER_LABEL].lower())
        # Remove double spaces.
        csv_entry[CSV_HEADER_LABEL] = csv_entry[CSV_HEADER_LABEL].strip().replace('  ', ' ')

    # Filter out labels that are only shorter than 2 characters.
    csv_data = list(filter(lambda x: len(x[CSV_HEADER_LABEL]) >= 2, csv_data))

    # Write list to CSV file.
    print('> Writing {} lines of {} files to {}'.format(len(csv_data), target, target_csv_path))
    # Delete the old file if it exists.
    storage.delete_file_if_exists(target_csv_path)

    # Write data to the file.
    with open(target_csv_path, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, delimiter=CSV_DELIMITER, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        writer.writerows(csv_data)

    return target_csv_path


def sort_csv_by_seq_len(csv_path, num_buckets=64, max_length=17.0):
    """
    Sort a train.csv like file by it's audio files sequence length.
    Additionally outputs longer than `max_length` are being discarded from the given CSV file.
    Also it prints out optimal bucket sizes after computation.

    Args:
        csv_path (str):
            Path to the `train.csv`.

        num_buckets (int):
            Number ob buckets to split the input into.

        max_length (float):
            Positive float. Maximum length in seconds for a feature vector to keep.
            Set to `0.` to keep everything.

    Returns:
        Tuple[List[int], float]:
            A tuple containing the boundary array and the total corpus length in seconds.
    """

    # Read train.csv file.
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=CSV_DELIMITER, fieldnames=CSV_FIELDNAMES)

        # Read all lines into memory and remove CSV header.
        csv_data = [csv_entry for csv_entry in reader][1:]

    # Sort entries by sequence length.
    csv_data = sorted(csv_data, key=lambda x: x[CSV_HEADER_LENGTH])

    # Remove samples longer than `max_length` points.
    if max_length > 0:
        number_of_entries = len(csv_data)
        csv_data = [d for d in csv_data if float(d[CSV_HEADER_LENGTH]) < max_length]
        print('Removed {:,d} examples because they are too long.'
              .format(number_of_entries - len(csv_data)))

    # Calculate optimal bucket sizes.
    lengths = [int(float(d[CSV_HEADER_LENGTH]) / WIN_STEP) for d in csv_data]
    step = len(lengths) // num_buckets

    buckets = set()
    for i in range(step, len(lengths), step):
        buckets.add(lengths[i])
    buckets = list(buckets)
    buckets.sort()
    print('Suggested buckets: ', buckets)

    # Plot histogram of feature vector length distribution.
    __plot_sequence_lengths(lengths)

    # Determine total corpus length in seconds.
    total_length_seconds = sum(map(lambda x: float(x[CSV_HEADER_LENGTH]), csv_data))

    # Write CSV data back to file.
    storage.delete_file_if_exists(csv_path)
    with open(csv_path, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, delimiter=CSV_DELIMITER, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        writer.writerows(csv_data)

    with open(csv_path, 'r', encoding='utf-8') as f:
        print('Successfully sorted {} lines of {}'.format(len(f.readlines()), csv_path))

        return buckets, total_length_seconds


@pyplot_display
def __plot_sequence_lengths(plt, lengths):
    # Plot histogram of feature vector length distribution.
    fig = plt.figure()
    plt.hist(lengths, bins=50, facecolor='green', alpha=0.75, edgecolor='black', linewidth=0.9)
    plt.title('Sequence Length\'s Histogram')
    plt.ylabel('Count')
    plt.xlabel('Length')
    plt.grid(True)

    return fig


if __name__ == '__main__':
    # Path to `train.csv` file.
    _csv_path = os.path.join(CSV_DIR, 'train.csv')

    # Display dataset stats.
    sort_csv_by_seq_len(_csv_path)
