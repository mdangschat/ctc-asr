"""Utility to download corpus data, if necessary."""

import os
import sys
import requests
import tarfile
from tqdm import tqdm
from urllib.parse import urlparse

from python.util import storage
from python.dataset.config import CACHE_DIR


def maybe_download_batch(urls, md5s, cache_archives=True):
    # TODO Document

    for url, md5 in zip(urls, md5s):
        maybe_download(url, md5=md5, cache_archive=cache_archives)


def maybe_download(url, md5=None, cache_archive=True):
    """Downloads a tar.gz archive file if it's not cached. The archive gets extracted afterwards.
    It is advised to call `cleanup_cache()` after pre-processing to remove the cached extracted
    folder.

    Args:
        url (str):
            URL for dataset download.
        md5 (str):
            Checksum for optional integrity check or `None`.
        cache_archive (bool):
            `True` if the downloaded archive should be kept, `False` if it should be deleted.

    Returns:
        Nothing.
    """
    file_name = os.path.basename(urlparse(url).path)
    storage_path = os.path.join(CACHE_DIR, '{}'.format(file_name))

    # Download archive if necessary.
    if not os.path.isfile(storage_path):
        __dl_with_progress(url, storage_path)
    else:
        print('Using cached archive: {}'.format(storage_path))

    # Optional md5 integrity check.
    if md5:
        md5sum = storage.md5(storage_path)
        assert md5 == md5sum, 'Checksum does not match.'

    # Extract archive to cache directory.
    assert tarfile.is_tarfile(storage_path)
    print('Starting extraction of: {}'.format(storage_path))
    # with tarfile.open(name=storage_path, mode='r') as tf:    TODO remove if other way works
    #     tf.extractall(path=CACHE_DIR)
    storage.tar_extract_all(storage_path, CACHE_DIR)
    print('Completed extraction of: {}'.format(storage_path))

    # Delete cached archive if requested.
    if not cache_archive:
        storage.delete_file_if_exists(storage_path)
        print('Cache file "{}" deleted.'.format(storage_path))


def cleanup_cache(directory_name):
    """

    Args:
        directory_name (str): Directory name of the extracted folder in the cache folder.
            This is NOT the folder's path.

    Returns:
        Nothing.
    """
    path = os.path.join(CACHE_DIR, directory_name)
    storage.delete_directory_if_exists(path)

    if not os.path.exists(path):
        print('Removed cached folder: {}'.format(path))
    else:
        print('WARN: Could not remove cached folder: {}'.format(path))


def __dl_with_progress(url, storage_path):
    r = requests.get(url, stream=True)
    content_length = int(r.headers.get('content-length'))
    chunk_size = 1024

    print('Starting to download "{}" ({:.3f} GiB) to: {}'
          .format(url, content_length / (1024 ** 3), storage_path), flush=True)

    with open(storage_path, 'wb') as f:
        pbar = tqdm(total=content_length, unit='iB', unit_scale=True, unit_divisor=1024,
                    file=sys.stdout)
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:  # Filter out keep-alive chunks.
                pbar.update(len(chunk))
                f.write(chunk)

        f.flush()
        pbar.close()
    print('Download finished.')


# For testing purposes.
if __name__ == '__main__':
    cv_tar = 'https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz'

    maybe_download(cv_tar, cache_archive=True)

    print('Dummy pre-processing here...')

    cleanup_cache('cv_corpus_v1')

    print('\nDone.')
