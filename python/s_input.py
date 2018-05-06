"""Routines to load a corpus and perform the necessary pre processing on the
audio files and labels.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow import contrib as tfc

from python.params import FLAGS, TF_FLOAT
import python.s_labels as s_labels
from python.loader.load_sample import load_sample, NUM_MFCC


NUM_INPUTS = NUM_MFCC * 2
DATA_PATH = '/home/marc/workspace/speech/data'


def inputs_train(batch_size, scope='train_input', txt_file='train.txt'):
    """Construct input for speech training.

    Args:
        batch_size (int):
            (Maximum) number of samples per batch.
            See: _generate_batch() and `allow_smaller_final_batch=True`
        scope (str):
            TensorFlow scope name.
        txt_file (str):
            Training filename.

    Returns:
        tf.Tensor:
            3D Tensor with sequence batch of shape [batch_size, time, data].
            Where time is equal to max(seq_len) for the bucket batch.
        tf.Tensor:
            1D Tensor with sequence lengths for each sequence within the batch.
            With shape [batch_size], and type tf.int32.
        tf.Tensor:
            2D Tensor with labels batch of shape [batch_size, max_label_len],
            with max_label_len equal to max(len(label)) for the bucket batch.
            Type is tf.int32.
        tf.Tensor:
            2D Tensor with the original strings.
    """
    # Info: Longest label list in TIMIT train/test is 79 characters long.
    train_txt_path = os.path.join(DATA_PATH, txt_file)
    sample_list, label_list, original_list = _read_file_list(train_txt_path)

    with tf.name_scope(scope):
        # Convert lists to tensors.
        file_names = tf.convert_to_tensor(sample_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.string)
        originals = tf.convert_to_tensor(original_list, dtype=tf.string)

        # Ensure that the random shuffling has good mixing properties.
        capacity = 512 + 4 * FLAGS.batch_size

        # Create an input queue that produces the file names to read.
        sample_queue, label_queue, originals_queue = tf.train.slice_input_producer(
            [file_names, labels, originals],
            capacity=capacity,
            num_epochs=None,
            shuffle=True,
            seed=FLAGS.random_seed
        )

        # Reinterpret the bytes of a string as a vector of numbers.
        label_queue = tf.decode_raw(label_queue, tf.int32)

        # Read the sample from disk and extract it's features.
        sample, sample_len = tf.py_func(load_sample, [sample_queue], [TF_FLOAT, tf.int32],
                                        name='py_load_sample')

        # Restore shape, since `py_func` forgets it.
        # See: https://www.tensorflow.org/api_docs/python/tf/Tensor#set_shape
        sample.set_shape([None, NUM_INPUTS])
        sample_len.set_shape([])    # Shape for scalar is [].

        print('Generating training batches of size {}. Queue capacity is {}. '
              .format(batch_size, capacity))

        sequences, seq_length, labels, originals = _generate_batch(
            sample, sample_len, label_queue, originals_queue, batch_size, capacity)

        # Convert the dense labels to sparse ones for the CTC loss function.
        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/dense_to_sparse
        labels = tfc.layers.dense_to_sparse(labels)

        return sequences, seq_length, labels, originals


def inputs(batch_size, scope='test_input', txt_file='test.txt'):
    """Construct input for speech evaluation. This method always returns unaltered data.

    Args:
        batch_size (int):
            (Maximum) number of samples per batch.
            See: _generate_batch() and `allow_smaller_final_batch=True`
        scope (str):
            TensorFlow scope name.
        txt_file (str):
            Training filename.

    Returns:
        tf.Tensor:
            3D Tensor with sequence batch of shape [batch_size, time, data].
            Where time is equal to max(seq_len) for the bucket batch.
        tf.Tensor:
            1D Tensor with sequence lengths for each sequence within the batch.
            With shape [batch_size], and type tf.int32.
        tf.Tensor:
            2D Tensor with labels batch of shape [batch_size, max_label_len],
            with max_label_len equal to max(len(label)) for the bucket batch.
            Type is tf.int32.
        tf.Tensor:
            2D Tensor with the original strings.
    """
    return inputs_train(batch_size, scope=scope, txt_file=txt_file)


def _read_file_list(path):
    """Generate synchronous lists of all samples with their respective lengths and labels.
    Labels are converted from characters to integers.
    See: `s_labels`.

    Args:
        path (str):
            Path to the training or testing .TXT files, e.g. `/some/path/timit/train.txt`

    Returns:
        [str]: List of absolute paths to .WAV files.
        [[int]]: List of labels filtered and converted to integer lists.
        [str]: List of original strings.
    """
    with open(path) as f:
        lines = f.readlines()

        sample_paths = []
        labels = []
        originals = []
        for line in lines:
            sample_path, label = line.split(' ', 1)
            sample_paths.append(os.path.join(DATA_PATH, 'timit/TIMIT', sample_path))
            label = label.strip()
            originals.append(label)
            label = [s_labels.ctoi(c) for c in label]
            label = np.array(label, dtype=np.int32).tostring()
            labels.append(label)

        return sample_paths, labels, originals


def _generate_batch(sequence, seq_len, label, original, batch_size, capacity):
    """Construct a queued batch of sample sequences and labels.

    Args:
        sequence (tf.Tensor):
            2D tensor of shape [time, NUM_INPUTS] with type float.
        seq_len (tf.Tensor):
            1D tensor of shape [1] with type int32.
        label (tf.Tensor):
            1D tensor of shape [<length label>] with type int32.
        original (tf.Tensor):
            1D tensor of shape [<length original text>] with type tf.string.
            The original text.
        batch_size (int):
            (Maximum) number of samples per batch.
        capacity (int):
            The maximum number of minibatches in the top queue,
            and also the maximum number of elements within each bucket.

    Returns:
        tf.Tensor:
            3D Tensor with sequence batch of shape [batch_size, time, data].
            Where time is equal to max(seq_len) for the bucket batch.
        tf.Tensor:
            1D Tensor with sequence lengths for each sequence within the batch.
            With shape [batch_size], and type tf.int32.
        tf.Tensor:
            2D Tensor with labels batch of shape [batch_size, max_label_len],
            with max_label_len equal to max(len(label)) for the bucket batch.
            Type is tf.int32.
        tf.Tensor:
            2D Tensor with the original strings.
    """
    num_threads = 8
    boundaries = [91, 130, 176, 228, 276, 313, 348, 387, 428, 469, 502, 527, 548, 566, 582, 597,
                  611, 625, 642, 916]

    # https://www.tensorflow.org/api_docs/python/tf/contrib/training/bucket_by_sequence_length
    seq_length, (sequences, labels, originals) = tfc.training.bucket_by_sequence_length(
        input_length=seq_len,
        tensors=[sequence, label, original],
        batch_size=batch_size,
        bucket_boundaries=boundaries,
        num_threads=num_threads,
        capacity=capacity // len(boundaries),
        # Pads smaller batch elements (sequence and label) to the size of the longest one.
        dynamic_pad=True,
        allow_smaller_final_batch=False
    )

    # Add input vectors to TensorBoard summary.
    batch_size_t = tf.shape(sequences)[0]
    summary_batch = tf.reshape(sequences, [batch_size_t, -1, NUM_INPUTS, 1])
    tf.summary.image('sequence', summary_batch, max_outputs=1)

    return sequences, seq_length, labels, originals
