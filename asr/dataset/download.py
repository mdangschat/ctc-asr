"""
Utility to download corpus data, if necessary.
"""

import os
import sys
import tarfile
import zipfile
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from asr.dataset.config import CACHE_DIR
from asr.util import storage


def maybe_download_batch(urls, md5s, cache_archives=True):
    """
    Download and extract a batch of archives.

    Args:
        urls (List[str]): List of download URLs.
        md5s (List[str]): List of MD5 checksums.
        cache_archives (bool): Keep the downloaded archives after extraction?

    Returns:
        Nothing.
    """

    for url, md5 in zip(urls, md5s):
        maybe_download(url, md5=md5, cache_archive=cache_archives)


def maybe_download(url, md5=None, cache_archive=True):
    """
    Downloads a archive file if it's not cached. The archive gets extracted afterwards.
    It is advised to call `cleanup_cache()` after pre-processing to remove the cached extracted
    folder.
    Currently only TAR and ZIP files are supported.

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
        download_with_progress(url, storage_path)
    else:
        print('Using cached archive: {}'.format(storage_path))

    # Optional md5 integrity check.
    if md5:
        md5sum = storage.md5(storage_path)
        assert md5 == md5sum, 'Checksum does not match.'

    # Extract archive to cache directory.
    print('Starting extraction of: {}'.format(storage_path))
    if tarfile.is_tarfile(storage_path):
        storage.tar_extract_all(storage_path, CACHE_DIR)
    elif zipfile.is_zipfile(storage_path):
        with zipfile.ZipFile(storage_path, 'r') as zip_:
            zip_.extractall(CACHE_DIR)
    else:
        raise ValueError('Compression method not supported: ', storage_path)
    print('Completed extraction of: {}'.format(storage_path))

    # Delete cached archive if requested.
    if not cache_archive:
        storage.delete_file_if_exists(storage_path)
        print('Cache file "{}" deleted.'.format(storage_path))


def cleanup_cache(directory_name):
    """
    Remove the given directory name from the projects `cache` directory.

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


def download_with_progress(url, storage_path):
    """
    Download a given `url` to the `storage_path` and display a progressbar.

    Args:
        url (str): The URL to download.
        storage_path (str): Where to store the download, e.g. `/tmp/archive.tar.gz`.

    Returns:
        Nothing.
    """

    request = requests.get(url, stream=True)
    content_length = int(request.headers.get('content-length'))
    chunk_size = 1024

    print('Starting to download "{}" ({:.3f} GiB) to: {}'
          .format(url, content_length / (1024 ** 3), storage_path), flush=True)

    with open(storage_path, 'wb') as file_handle:
        pbar = tqdm(total=content_length, unit='iB', unit_scale=True, unit_divisor=1024,
                    file=sys.stdout)
        for chunk in request.iter_content(chunk_size=chunk_size):
            if chunk:  # Filter out keep-alive chunks.
                pbar.update(len(chunk))
                file_handle.write(chunk)

        file_handle.flush()
        pbar.close()
    print('Download finished.')


# For testing purposes.
if __name__ == '__main__':
    __TAR_ARCHIVE = 'https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz'

    maybe_download(__TAR_ARCHIVE, cache_archive=True)

    print('Dummy pre-processing here...')

    cleanup_cache('cv_corpus_v1')

    print('\nDone.')
