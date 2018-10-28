"""Utility to download corpus data, if necessary."""

import os
import sys
import requests
import tarfile
from tqdm import tqdm
from urllib.parse import urlparse

from asr.params import BASE_PATH
from asr.util import storage

# Cache folder.
__CACHE = os.path.join(BASE_PATH, 'data/cache')


def maybe_download(url, cache_archive=True):
    # TODO Documentation
    # Downloads a tar.gz archive file if it's not cached. The archive gets extracted afterwards.

    # TODO This should not be done in here, the individual dataset wrapper prepares the data

    # TODO Wrapper for all used dataset wrappers, that creates the train.txt, test.txt, dev.txt

    # TODO My current preference is to keep the .tar.gz file and delete the extracted data after
    # processing it.

    file_name = os.path.basename(urlparse(url).path)
    storage_path = os.path.join(__CACHE, '{}'.format(file_name))

    # Download archive if necessary.
    # L8ER Add optional MD5 check
    if not os.path.isfile(storage_path):
        __dl_with_progress(url, storage_path)
    else:
        print('Using cached archive: {}'.format(storage_path))

    # Extract archive to cache directory.
    assert tarfile.is_tarfile(storage_path)
    print('Starting extraction of: {}'.format(storage_path))
    with tarfile.open(name=storage_path, mode='r') as tf:
        tf.extractall(path=__CACHE)
        print('Completed extraction to: {}'.format(__CACHE))

    # Delete cached archive if requested.
    if not cache_archive:
        storage.delete_file_if_exists(storage_path)
        print('Cache file "{}" deleted.'.format(storage_path))


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


# TODO delete afterwards, only for testing.
if __name__ == '__main__':
    dummy_tar = 'https://osdn.net/frs/g_redir.php?m=kent&f=od1n%2Fsamples.tar.gz'
    cv_tar = 'https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz'

    maybe_download(cv_tar, cache_archive=True)

    print('\nDone.')
