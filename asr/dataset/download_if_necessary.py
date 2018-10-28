"""Utility to download corpus data, if necessary."""

import os
import sys
import requests
from tqdm import tqdm
from urllib.parse import urlparse

from asr.params import BASE_PATH


def dl_if_necessary(url, cache_archive=True):
    # TODO Documentation

    # TODO Define cache folder
    # TODO Check if archive is cached, else download it
    # TODO Extract archive
    # TODO This should not be done in here, the individual dataset wrapper prepares the data
    # TODO Wrapper for all used dataset wrappers, that creates the train.txt, test.txt, dev.txt

    file_name = os.path.basename(urlparse(url).path)
    path = '/tmp/{}'.format(file_name)

    __dl_with_progress(url, path)


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
    dummy_1_100 = 'https://speed.hetzner.de/100MB.bin'
    dummy_2_100 = 'http://ipv4.download.thinkbroadband.com/100MB.zip'
    dummy_3_100 = 'http://www.ovh.net/files/100Mb.dat'
    dummy_1 = 'https://speed.hetzner.de/1GB.bin'

    dl_if_necessary(dummy_1_100, cache_archive=False)

    print('\nDone.')
