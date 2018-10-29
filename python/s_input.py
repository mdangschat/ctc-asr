"""Routines to load a corpus and perform the necessary pre processing on the audio files and labels.
"""

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import contrib as tfc

from python.params import FLAGS, TF_FLOAT, BASE_PATH
import python.labels as s_labels
from python.load_sample import load_sample, NUM_FEATURES


# Path to train.txt file.
TRAIN_TXT_PATH = os.path.join(BASE_PATH, 'data/train.txt')
# Path to train.txt file.
TEST_TXT_PATH = os.path.join(BASE_PATH, 'data/test.txt')
# Path to dev.txt file.
DEV_TXT_PATH = os.path.join(BASE_PATH, 'data/dev.txt')
# Path to dataset collection folder.
DATASET_PATH = os.path.join(BASE_PATH, '../datasets/speech_data/')


def inputs_train(batch_size, shuffle=False):
    """Construct input for asr training.

    Args:
        batch_size (int):
            (Maximum) number of samples per batch.
            See: _generate_batch() and `allow_smaller_final_batch=True`

        shuffle (bool):
            Default (`False`) create batches from samples in the order they are listed in
            the .txt file. Else (`True`) shuffle input order. This also uses bucketing, to
            only combine samples of similar sequence length into a batch.

    Returns:
        tf.Tensor: `sequences`
            3D Tensor with sequence batch of shape [batch_size, time, data].
            Where time is equal to max(seq_len) for the bucket batch.
        tf.Tensor: `seq_len`
            1D Tensor with sequence lengths for each sequence within the batch.
            With shape [batch_size], and type tf.int32.
        tf.Tensor: `labels`
            2D Tensor with labels batch of shape [batch_size, max_label_len],
            with max_label_len equal to max(len(label)) for the bucket batch.
            Type is tf.int32.
        tf.Tensor: `label_len`
            tf.int32 Tensor, containing the length of each of the labels within the batch.
        tf.Tensor: `originals`
            2D Tensor with the original strings.
    """
    sample_list, label_list, original_list = _read_file_list(TRAIN_TXT_PATH)

    with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
        # Convert lists to tensors.
        file_names = tf.convert_to_tensor(sample_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.string)
        originals = tf.convert_to_tensor(original_list, dtype=tf.string)

        # Set a sufficient bucket capacity.
        capacity = 512 + (4 * FLAGS.batch_size)

        # Create an input queue that produces the file names to read.
        __random_seed = FLAGS.random_seed if FLAGS.random_seed != 0 else int(time.time())
        sample_queue, label_queue, originals_queue = tf.train.slice_input_producer(
            [file_names, labels, originals],
            capacity=capacity,
            num_epochs=1,
            shuffle=shuffle,
            seed=__random_seed
        )

        # Reinterpret the bytes of a string as a vector of numbers.
        label_queue = tf.decode_raw(label_queue, tf.int32)

        # Determine length of the label vector.
        label_len = tf.shape(label_queue)

        # Read the sequence from disk and extract its features.
        sequence, seq_len = tf.py_func(load_sample, [sample_queue], [TF_FLOAT, tf.int32],
                                       name='py_load_sample')

        # Restore shape, since `py_func` forgets it.
        # See: https://www.tensorflow.org/api_docs/python/tf/Tensor#set_shape
        sequence.set_shape([None, NUM_FEATURES])
        seq_len.set_shape([])    # Shape for scalar is [].

        print('Generating training batches of size {}. Queue capacity is {}.'
              .format(batch_size, capacity))

        if shuffle:
            batch = _generate_bucket_batch(sequence, seq_len, label_queue, label_len,
                                           originals_queue, batch_size, capacity)
        else:
            batch = _generate_sorted_batch(sequence, seq_len, label_queue, label_len,
                                           originals_queue, batch_size)

        sequences, seq_length, labels, label_len, originals = batch

        # Convert the dense labels to sparse ones for the CTC-loss function.
        if not FLAGS.use_warp_ctc:
            # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/dense_to_sparse
            labels = tfc.layers.dense_to_sparse(labels)

        # Add input vectors to TensorBoard summary.
        batch_size_t = tf.shape(sequences)[0]
        summary_batch = tf.reshape(sequences, [batch_size_t, -1, NUM_FEATURES, 1])
        tf.summary.image('sequence', summary_batch, max_outputs=1)

        return sequences, seq_length, labels, label_len, originals


def inputs(batch_size, target):
    """Construct input for asr evaluation. This method always returns unaltered data.

    Args:
        batch_size (int):
            (Maximum) number of samples per batch.
            See: _generate_batch() and `allow_smaller_final_batch=True`
        target (str):
            Which dataset to use. Supported: 'test' or 'dev'.

    Returns:
        tf.Tensor: `sequences`
            3D Tensor with sequence batch of shape [batch_size, time, data].
            Where time is equal to max(seq_len) for the bucket batch.
        tf.Tensor: `seq_len`
            1D Tensor with sequence lengths for each sequence within the batch.
            With shape [batch_size], and type tf.int32.
        tf.Tensor: `labels`
            2D Tensor with labels batch of shape [batch_size, max_label_len],
            with max_label_len equal to max(len(label)) for the bucket batch.
            Type is tf.int32.
        tf.Tensor: `label_len`
            tf.int32 Tensor, containing the length of each of the labels within the batch.
        tf.Tensor: `originals`
            2D Tensor with the original strings.
    """
    if target == 'test':
        txt_path = TEST_TXT_PATH
    elif target == 'dev':
        txt_path = DEV_TXT_PATH
    else:
        raise ValueError('Invalid target "{}".'.format(target))
    print('Using: {} as input.'.format(txt_path))

    paths_list, labels_list, originals_list = _read_file_list(txt_path)

    with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
        # Convert lists to tensors.
        paths_t = tf.convert_to_tensor(paths_list, dtype=tf.string)
        labels_t = tf.convert_to_tensor(labels_list, dtype=tf.string)
        originals_t = tf.convert_to_tensor(originals_list, dtype=tf.string)

        # Set a sufficient bucket capacity.
        capacity = 256 + (4 * FLAGS.batch_size)

        # Create an input queue that produces the file names to read.
        __random_seed = FLAGS.random_seed if FLAGS.random_seed != 0 else int(time.time())
        path_queue, label_queue, original_queue = tf.train.slice_input_producer(
            [paths_t, labels_t, originals_t],
            capacity=capacity,
            num_epochs=1,
            shuffle=True,
            seed=__random_seed
        )

        # Reinterpret the bytes of a string as a vector of numbers.
        label_queue = tf.decode_raw(label_queue, tf.int32)

        # Determine length of the label vector.
        label_len = tf.shape(label_queue)

        # Read the sequence from disk and extract its features.
        sequence, seq_len = tf.py_func(load_sample, [path_queue], [TF_FLOAT, tf.int32],
                                       name='py_load_sample')

        # Restore shape, since `py_func` forgets it.
        # See: https://www.tensorflow.org/api_docs/python/tf/Tensor#set_shape
        sequence.set_shape([None, NUM_FEATURES])
        seq_len.set_shape([])  # Shape for scalar is [].

        print('Generating validation batches of size {}. Queue capacity is {}.'
              .format(batch_size, capacity))

        batch = _generate_bucket_batch(sequence, seq_len, label_queue, label_len, original_queue,
                                       batch_size, capacity)

        sequences, seq_length, labels_t, label_len, originals_t = batch

        # Convert the dense labels to sparse ones for the CTC-loss function.
        if not FLAGS.use_warp_ctc:
            # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/dense_to_sparse
            labels_t = tfc.layers.dense_to_sparse(labels_t)

        # Add input vectors to TensorBoard summary.
        batch_size_t = tf.shape(sequences)[0]
        summary_batch = tf.reshape(sequences, [batch_size_t, -1, NUM_FEATURES, 1])
        tf.summary.image('sequence', summary_batch, max_outputs=1)

        return sequences, seq_length, labels_t, label_len, originals_t


def _generate_sorted_batch(sequence, seq_len, label, label_len, original, batch_size):
    """Construct a queued batch of sample sequences and labels.

    Args:
        sequence (tf.Tensor):
            2D tensor of shape [time, NUM_INPUTS] with type float.
        seq_len (tf.Tensor):
            1D tensor of shape [1] with type int32.
        label (tf.Tensor):
            1D tensor of shape [<length label>] with type int32.
        label_len (tf.Tensor):
            tf.int32 Tensor, containing the length of the `label` Tensor.
        original (tf.Tensor):
            1D tensor of shape [<length original text>] with type tf.string.
            The original text.
        batch_size (int):
            Number of samples per batch.

    Returns:
        tf.Tensor: `sequences`
            3D Tensor with sequence batch of shape [batch_size, time, data].
            Where time is equal to max(seq_len) for the bucket batch.
        tf.Tensor: `seq_len`
            1D Tensor with sequence lengths for each sequence within the batch.
            With shape [batch_size], and type tf.int32.
        tf.Tensor: `labels`
            2D Tensor with labels batch of shape [batch_size, max_label_len],
            with max_label_len equal to max(len(label)) for the bucket batch.
            Type is tf.int32.
        tf.Tensor: `label_len`
            tf.int32 Tensor, containing the length of each of the labels within the batch.
        tf.Tensor: `originals`
            2D Tensor with the original strings.
    """
    sequences, seq_len, labels, label_len, originals = tf.train.batch(
        tensors=[sequence, seq_len, label, label_len, original],
        batch_size=batch_size,
        num_threads=FLAGS.num_threads,
        capacity=128,
        enqueue_many=False,
        shapes=None,
        dynamic_pad=True,
        allow_smaller_final_batch=False
    )
    return sequences, seq_len, labels, label_len, originals


def _generate_bucket_batch(sequence, seq_len, label, label_len, original, batch_size, capacity):
    """Construct a queued batch of sample sequences and labels using buckets.

    Args:
        sequence (tf.Tensor):
            2D tensor of shape [time, NUM_INPUTS] with type float.
        seq_len (tf.Tensor):
            1D tensor of shape [1] with type int32.
        label (tf.Tensor):
            1D tensor of shape [<length label>] with type int32.
        label_len (tf.Tensor):
            tf.int32 Tensor, containing the length of the `label` Tensor.
        original (tf.Tensor):
            1D tensor of shape [<length original text>] with type tf.string.
            The original text.
        batch_size (int):
            Number of samples per batch.
        capacity (int):
            The maximum number of mini-batches in the top queue,
            and also the maximum number of elements within each bucket.

    Returns:
        tf.Tensor: `sequences`
            3D Tensor with sequence batch of shape [batch_size, time, data].
            Where time is equal to max(seq_len) for the bucket batch.
        tf.Tensor: `seq_len`
            1D Tensor with sequence lengths for each sequence within the batch.
            With shape [batch_size], and type tf.int32.
        tf.Tensor: `labels`
            2D Tensor with labels batch of shape [batch_size, max_label_len],
            with max_label_len equal to max(len(label)) for the bucket batch.
            Type is tf.int32.
        tf.Tensor: `label_len`
            tf.int32 Tensor, containing the length of each of the labels within the batch.
        tf.Tensor: `originals`
            2D Tensor with the original strings.
    """
    if FLAGS.features_drop_every_second_frame:
        boundaries = [143, 154, 161, 168, 172, 179, 183, 186, 193, 197, 203, 204, 208, 211, 212,
                      219, 222, 229, 233, 237, 244, 247, 251, 255, 258, 263, 273, 276, 283, 294,
                      300, 311, 323, 337, 353, 373, 400, 431, 472, 517, 575, 640, 708, 772, 834,
                      901, 975, 1056, 1136, 1207, 1266, 1315, 1357, 1393, 1425, 1455, 1484, 1512,
                      1540, 1568, 1600]
    else:
        boundaries = [71, 77, 80, 84, 86, 89, 91, 93, 96, 98, 101, 102, 104, 105, 106, 109, 111,
                      114, 116, 118, 122, 123, 125, 127, 129, 131, 136, 138, 141, 147, 150, 155,
                      161, 168, 176, 186, 200, 215, 236, 258, 287, 320, 354, 386, 417, 450, 487,
                      528, 568, 603, 633, 657, 678, 696, 712, 727, 742, 756, 770, 784, 800, 849]

    # https://www.tensorflow.org/api_docs/python/tf/contrib/training/bucket_by_sequence_length
    seq_len, (sequences, labels, label_len, originals) = \
        tfc.training.bucket_by_sequence_length(
        input_length=seq_len,
        tensors=[sequence, label, label_len, original],
        batch_size=batch_size,
        bucket_boundaries=boundaries,
        num_threads=FLAGS.num_threads,
        capacity=capacity // len(boundaries),
        # Pads smaller batch elements (sequence and label) to the size of the longest one.
        dynamic_pad=True,
        allow_smaller_final_batch=False
    )
    return sequences, seq_len, labels, label_len, originals


def _read_file_list(txt_path):
    """Generate synchronous lists of all samples with their respective lengths and labels.
    Labels are converted from characters to integers.

    Also see `s_labels` documentation.

    Args:
        txt_path (str):
            Path to the training or testing .TXT files, e.g. `/some/path/train.txt`

    Returns:
        [str]: List of absolute paths to .WAV files.
        [[int]]: List of labels filtered and converted to integer lists.
        [str]: List of original strings.
    """
    with open(txt_path) as f:
        lines = f.readlines()

        paths = []
        labels = []
        originals = []
        for line in lines:
            path, label = line.split(' ', 1)
            paths.append(os.path.join(DATASET_PATH, path))

            originals.append(label)

            label = [s_labels.ctoi(c) for c in label.strip()]
            label = np.array(label, dtype=np.int32).tostring()
            labels.append(label)

        return paths, labels, originals
