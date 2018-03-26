"""Routines to load the traffic sign (TS) corpus [BelgiumTS (cropped images)]
and transform the images into an usable format.

.. _BelgiumTS:
   http://btsd.ethz.ch/shareddata/

"""

import os
import tensorflow as tf

from loader import load_input


NUMBER_CLASSES = 62     # TODO
INPUT_SHAPE = ()        # TODO
SAMPLING_RATE = 16000
NUM_EXAMPLES_PER_EPOCH_TRAIN = 4620
NUM_EXAMPLES_PER_EPOCH_EVAL = 1680
DATA_PATH = '/home/marc/workspace/speech/data'

FLAGS = tf.app.flags.FLAGS


def inputs_train(data_dir, batch_size):
    """Construct input for TS training.
    review Documentation

    Args:
        data_dir: Path to the data directory.
        batch_size (int): Number of images per batch.

    Returns:
        images: Images a 4D tensor of
                [batch_size, IMAGE_SHAPE[0], IMAGE_SHAPE[1], INPUT_SHAPE[2]] size.
        labels: Labels a 1D tensor of [batch_size] size.
    """
    train_txt_path = os.path.join(data_dir, 'train.txt')
    sample_list, label_list = _read_file_list(train_txt_path)

    with tf.name_scope('train_input'):
        # Convert lists to tensors.
        file_names = tf.convert_to_tensor(sample_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.string)
        print('train_input:', file_names, labels)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.33
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_TRAIN * min_fraction_of_examples_in_queue)
        capacity = min_queue_examples + 3 * batch_size

        # Create an input queue that produces the file names to read.
        sample_queue, label_queue = tf.train.slice_input_producer(
            [file_names, labels], capacity=capacity, num_epochs=None, shuffle=False)

        print('queues:', sample_queue, label_queue)

        # TODO write read_sample()
        # review: The body of the function (i.e. func) will not be serialized in a GraphDef.
        # Therefore, you should not use this function if you need to serialize your model and
        # restore it in a different environment.
        sample, label = tf.py_func(_read_sample,
                                   [sample_queue, label_queue],
                                   [tf.string, tf.string])

        print('py_func:', sample, label)

        print('Filling the queue with {} images before starting to train. '
              'Queue capacity is {}. '
              'This will take a few minutes.'
              .format(min_queue_examples, capacity))

    return _generate_batch(sample, label, min_queue_examples, batch_size, shuffle=True)


def inputs(eval_data, data_dir, image_shape, batch_size):
    # TODO: Rewrite this function to match inputs_train().
    """Construct fitting input for the evaluation process.

    Args:
        eval_data (boolean): Indicating if one should use the train or eval data set.
        data_dir (str): Path to the data directory.
        image_shape ([int]): Three element shape array.
                     E.g. [32, 32, 3] for colored images and [32, 32, 1] for monochrome images.
        batch_size (int): Number of images per batch.

    Returns:
        images: Images a 4D tensor of [batch_size, IMAGE_SHAPE[0], IMAGE_SHAPE[1], INPUT_SHAPE[2]]
                size.
        labels: Labels a 1D tensor of [batch_size] size.
    """
    if not eval_data:   # review Is this really needed? Makes everything more complicated.
        data_path = os.path.join(data_dir, 'Training')
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_TRAIN
    else:
        data_path = os.path.join(data_dir, 'Testing')
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_EVAL

    raise NotImplementedError


def _read_sample(sample_queue, label_queue):
    """Reads and converts the TS data files.
    review Documentation

    Args:
        sample_queue: A TensorFlow queue of tuples with the file names to read from and labels.
                     Compare: tf.train.slice_input_producer
        label_queue: Numpy like shape with 3 elements. For example:
                     [32, 32, 3] for colored images and [32, 32, 1] for monochrome images.

    Returns:
        reshaped_image: A single example.
        label: The corresponding label.
    """
    sample = load_input.load_sample(sample_queue, expected_sr=SAMPLING_RATE)

    label = label_queue

    print('Reshaped sample tensor:', sample, type(sample))
    print('Label:', label, type(label))

    return sample, label


def _read_file_list(path):
    """Generate two synchronous lists of all image samples and their respective labels
    within the provided path.
    review Documentation

    Args:
        path (str): Path to the training or testing folder of the TS data set.

    Returns:
        file_names ([str]): A list of file name strings.
        labels ([int]): A list of labels as integer.

    .. _StackOverflow:
       https://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
    """
    print('Opening ', path)     # todo remove
    with open(path) as f:
        lines = f.readlines()

        samples = []
        labels = []
        for line in lines:
            sample, label = line.split(' ', 1)
            samples.append(os.path.join(DATA_PATH, 'timit', sample))
            labels.append(label.strip())

        return samples, labels


def _generate_batch(sample, label, min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of images and labels.
    review Documentation

    Args:
        sample: 3D tensor of [height, width, 1] of type float32.
        label: 1D tensor of type int32.
        min_queue_examples (int): Minimum number of samples to retain in the queue that
                                  provides the example batches.
        batch_size (int): Number of images per batch.
        shuffle (boolean): Indicating whether to use shuffling queue or not.

    Returns:
        images: Images 4D tensor of [batch_size, height, width, 1] size.
        labels: Labels 1D tensor of [batch_size] size.
    """
    num_pre_process_threads = 12
    capacity = min_queue_examples + 3 * batch_size

    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
            [sample, label],
            batch_size=batch_size,
            num_threads=num_pre_process_threads,
            capacity=capacity,
            min_after_dequeue=min_queue_examples
        )
    else:
        image_batch, label_batch = tf.train.batch(
            [sample, label],
            batch_size=batch_size,
            num_threads=num_pre_process_threads,
            capacity=capacity
        )

    # Display the training images in the visualizer.
    # tf.summary.image('images', image_batch, max_outputs=10)    # TODO: Summay options for audio?

    return image_batch, label_batch
