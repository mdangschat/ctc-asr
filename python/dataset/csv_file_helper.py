"""
Helper methods to generate the CSV files.
"""

import csv
import os
import re
import sys
from multiprocessing import Pool, Lock, cpu_count

from tqdm import tqdm

from python.dataset.config import CORPUS_DIR, CSV_DIR, CSV_DELIMITER
from python.dataset.config import LABEL_WHITELIST_PATTERN, CSV_HEADER_PATH, CSV_HEADER_LABEL
from python.load_sample import load_sample
from python.util import storage
from python.util.matplotlib_helper import pyplot_display


def generate_csv(dataset_name, target, csv_data):
    """
    Generate CSV files containing the audio path and the corresponding sentence.
    Generated files are being stored at `CSV_TARGET_PATH`.

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
        writer = csv.DictWriter(f, delimiter=CSV_DELIMITER,
                                fieldnames=[CSV_HEADER_PATH, CSV_HEADER_LABEL])
        writer.writeheader()

        writer.writerows(csv_data)

    return target_csv_path


def sort_csv_by_seq_len(csv_path, num_buckets=64, max_length=1700):
    """
    Sort a train.csv like file by it's audio files sequence length.
    Additionally outputs longer than `max_length` are being discarded from the given CSV file.
    Also it prints out optimal bucket sizes after computation.

    Args:
        csv_path (str):
            Path to the `train.csv`.

        num_buckets (int):
            Number ob buckets to split the input into.

        max_length (int):
            Positive integer. Max length for a feature vector to keep.
            Set to `0` to keep everything.

    Returns:
        Tuple[List[int], float]:
            A tuple containing the boundary array and the total corpus length in seconds.
    """
    # Read train.csv file.
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=CSV_DELIMITER,
                                fieldnames=[CSV_HEADER_PATH, CSV_HEADER_LABEL])

        # # Read all lines into memory. Remove CSV header.
        lines = [line for line in reader][1:]

        # TODO: REMOVE DEBUG STUFF
        entry = lines[0]
        print("LINES:", lines[:5])
        print("ENTRY:", entry)
        print("PATH:", entry[CSV_HEADER_PATH])
        print("LABEL:", entry[CSV_HEADER_LABEL])

        # Setup thread pool.
        lock = Lock()
        buffer = []  # Output buffer.

        with Pool(processes=cpu_count()) as pool:
            for result in tqdm(pool.imap_unordered(_feature_length_fn, lines, chunksize=4),
                               desc='Reading audio samples', total=len(lines), file=sys.stdout,
                               unit='samples', dynamic_ncols=True):
                lock.acquire()
                buffer.append(result)
                lock.release()

        # Sort by sequence length.
        buffer = sorted(buffer, key=lambda x: x[0])

        # Remove samples longer than `max_length` points.
        if max_length > 0:
            original_length = len(buffer)
            buffer = [s for s in buffer if s[0] < max_length]
            print('Removed {:,d} samples from training, because they are too long.'
                  .format(original_length - len(buffer)))

        # Calculate optimal bucket sizes.
        lengths = [l[0] for l in buffer]
        step = len(lengths) // num_buckets
        buckets = set()
        for i in range(step, len(lengths), step):
            buckets.add(lengths[i])
        buckets = list(buckets)
        buckets.sort()
        print('Suggested buckets: ', buckets)

        # Plot histogram of feature vector length distribution.
        _plot_sequence_lengths(lengths)

        # Determine total corpus length in seconds.
        total_length_seconds = sum(map(lambda x: x[0], buffer)) / 0.1

        # Remove sequence length from tuples, therefore, restoring the CSV entry list.
        csv_data = [csv_entry for _, csv_entry in buffer]

    # Write back to file.
    storage.delete_file_if_exists(csv_path)
    with open(csv_path, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, delimiter=CSV_DELIMITER,
                                fieldnames=[CSV_HEADER_PATH, CSV_HEADER_LABEL])
        writer.writeheader()

        writer.writerows(csv_data)

    with open(csv_path, 'r', encoding='utf-8') as f:
        print('Successfully sorted {} lines of {}'.format(len(f.readlines()), csv_path))

    return buckets[: -1], total_length_seconds


def _feature_length_fn(csv_entry):
    # Python multiprocessing helper method.

    length = int(load_sample(os.path.join(CORPUS_DIR, csv_entry[CSV_HEADER_PATH]))[1])
    return length, csv_entry


@pyplot_display
def _plot_sequence_lengths(plt, lengths):
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
