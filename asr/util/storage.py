"""Storage and version control helper methods."""

import os
import time

from git import Repo
import tensorflow as tf


def git_revision_hash():
    """Return the git revision id/hash.

    Returns:
        str: Git revision hash.
    """
    repo = Repo('.', search_parent_directories=True)
    return repo.head.object.hexsha


def git_branch():
    """Return the active git branches name.

    Returns:
        str: Git branch.
    """
    repo = Repo('.', search_parent_directories=True)
    try:
        branch_name = repo.active_branch.name
    except TypeError:
        branch_name = 'DETACHED HEAD'
    return branch_name


def git_latest_tag():
    """Return the latest added git tag.

    Returns:
        str: Git tag.
    """
    repo = Repo('.', search_parent_directories=True)
    tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
    return tags[-1].name


def delete_file_if_exists(path):
    """Delete the file for the given path, if it exists.

    Args:
        path (str): File path.

    Returns:
        Nothing.
    """
    if os.path.exists(path) and os.path.isfile(path):
        for i in range(5):
            try:
                os.remove(path)
                break
            except (OSError, ValueError) as e:
                print('TODO CAUGHT error at file: ', i, path)      # TODO: Debug
                print('ERROR=', type(e), e)                     # Debug
                if i == 4:
                    raise RuntimeError(path) from e
                time.sleep(1)


def maybe_delete_checkpoints(path, delete):
    """Delete a TensorFlow checkpoint directory if requested and necessary.

    Args:
        path (str):
            Path to directory e.g. `FLAGS.train_dir`.
        delete (bool):
            Whether to delete old checkpoints or not. Should probably correspond to `FLAGS.delete`.

    Returns:
        Nothing.
    """
    if tf.gfile.Exists(path) and delete:
        print('Deleting old checkpoint data from: {}'.format(path))
        tf.gfile.DeleteRecursively(path)
        tf.gfile.MakeDirs(path)
    elif tf.gfile.Exists(path) and not delete:
        print('Resuming training from: {}'.format(path))
    else:
        print('Starting a new training run in: {}'.format(path))
        tf.gfile.MakeDirs(path)
