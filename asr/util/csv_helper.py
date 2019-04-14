"""Helper methods to generate the CSV files."""

import csv
import os

from asr.params import CSV_HEADER_LENGTH, CSV_FIELDNAMES, CSV_DELIMITER, WIN_STEP


def get_bucket_boundaries(csv_path, num_buckets):
    """Generate a list of bucket boundaries, based on the example length in the CSV file.

    The boundaries are chose based on the distribution of example lengths, to allow each bucket
    to fill up at the same rate. This produces at max `num_buckets`.

    Args:
        csv_path (str): Path to the CSV file. E.g. '../data/train.csv'.
        num_buckets (int): The maximum amount of buckets to create.

    Returns:
        List[int]: List containing bucket boundaries.
    """
    assert os.path.exists(csv_path) and os.path.isfile(csv_path)

    with open(csv_path, 'r', encoding='utf-8') as file_handle:
        reader = csv.DictReader(file_handle, delimiter=CSV_DELIMITER, fieldnames=CSV_FIELDNAMES)
        csv_data = [csv_entry for csv_entry in reader][1:]

        # Calculate optimal bucket sizes.
        lengths = [int(float(d[CSV_HEADER_LENGTH]) / WIN_STEP) for d in csv_data]
        step = len(lengths) // num_buckets

        buckets = set()
        for i in range(step, len(lengths), step):
            buckets.add(lengths[i])
        buckets = list(buckets)
        buckets.sort()

        return buckets
