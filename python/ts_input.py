"""Routines to load the traffic sign (TS) corpus [BelgiumTS (cropped images)]
and transform the images into an usable format.

.. _BelgiumTS:
   http://btsd.ethz.ch/shareddata/

"""

import os

import numpy as np
from matplotlib import pyplot as plt
from skimage import data, transform
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod


INPUT_SHAPE = (64, 64, 3)
NUMBER_CLASSES = 62
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 4575
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2520
DATA_PATH = "/home/marc/workspace/tensorflow-tutorials/traffic_signs/data"

FLAGS = tf.app.flags.FLAGS


def inputs_train(data_dir, image_shape, batch_size):
    """Construct input for TS training.

    Args:
        data_dir: Path to the data directory.
        image_shape: Numpy like shape with 3 elements. For example:
                     [32, 32, 3] for colored images and [32, 32, 1] for monochrome images.
        batch_size (int): Number of images per batch.

    Returns:
        images: Images a 4D tensor of
                [batch_size, IMAGE_SHAPE[0], IMAGE_SHAPE[1], INPUT_SHAPE[2]] size.
        labels: Labels a 1D tensor of [batch_size] size.
    """
    data_path = os.path.join(data_dir, 'Training')
    image_list, labels_list = _gen_labeled_input_list(data_path)

    with tf.name_scope('train_input'):
        # Convert lists to tensors.
        file_names = tf.convert_to_tensor(image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(labels_list, dtype=tf.int32)

        # Create an input queue that produces the file names to read.
        input_queue = tf.train.slice_input_producer([file_names, labels],
                                                    num_epochs=None, shuffle=False)

        # Load the raw data.
        raw_images, raw_labels = _read_data(input_queue, image_shape)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.
        distorted_image = tf.image.random_brightness(raw_images, max_delta=56)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.3, upper=1.7)

        # Subtract of the mean and divide by the variance of the pixels.
        norm_images = tf.image.per_image_standardization(distorted_image)

        # Set the shapes of the images tensor.
        norm_images.set_shape(image_shape)

        # Convert labels and set the shape of the labels tensor.
        labels = tf.cast(raw_labels, tf.int32)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.33
        min_queue_examples = int(
            NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
        print('Filling the queue with {} images before starting to train. '
              'This will take a few minutes.'
              .format(min_queue_examples))

    return _generate_image_and_label_batch(norm_images, labels, min_queue_examples, batch_size,
                                           shuffle=True)


def inputs(eval_data, data_dir, image_shape, batch_size):
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
    if not eval_data:
        data_path = os.path.join(data_dir, 'Training')
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        data_path = os.path.join(data_dir, 'Testing')
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    image_list, labels_list = _gen_labeled_input_list(data_path)

    with tf.name_scope('eval_input'):
        # Convert lists to tensors.
        file_names = tf.convert_to_tensor(image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(labels_list, dtype=tf.int32)

        # Create an input queue that produces the file names to read.
        input_queue = tf.train.slice_input_producer([file_names, labels],
                                                    num_epochs=None, shuffle=False)

        # Load the raw data.
        raw_images, raw_labels = _read_data(input_queue, image_shape)

        # Subtract of the mean and divide by the variance of the pixels.
        norm_images = tf.image.per_image_standardization(raw_images)

        # Set the shapes of the images tensor.
        norm_images.set_shape(image_shape)

        # Convert labels and set the shape of the labels tensor.
        labels = tf.cast(raw_labels, tf.int32)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.33
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
        print('Filling the queue with {} images before starting to test. '
              'This will take a few minutes.'
              .format(min_queue_examples))

    return _generate_image_and_label_batch(norm_images, labels, min_queue_examples, batch_size,
                                           shuffle=False)


def _read_data(input_queue, image_shape):
    """Reads and converts the TS data files.

    Args:
        input_queue: A TensorFlow queue of tuples with the file names to read from and labels.
                     Compare: tf.train.slice_input_producer
        image_shape: Numpy like shape with 3 elements. For example:
                     [32, 32, 3] for colored images and [32, 32, 1] for monochrome images.

    Returns:
        reshaped_image: A single example.
        label: The corresponding label.
    """
    image_bytes = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_bytes)
    label = input_queue[1]

    # Optionally convert images to grayscale.
    if image_shape[2] == 1:
        image = tf.image.rgb_to_grayscale(image)

    # Convert image data from uint8 to float32.
    image_float = tf.cast(image, tf.float32)

    # Resize the images.
    resized_image = tf.image.resize_images(image_float, image_shape[:2],
                                           method=ResizeMethod.BILINEAR)

    # Reshape the images.
    reshaped_image = tf.reshape(resized_image, image_shape)
    print('Reshaped image tensor:', reshaped_image)

    return reshaped_image, label


def _gen_labeled_input_list(path):
    """Generate two synchronous lists of all image samples and their respective labels
    within the provided path.

    Args:
        path (str): Path to the training or testing folder of the TS data set.

    Returns:
        file_names ([str]): A list of file name strings.
        labels ([int]): A list of labels as integer.

    .. _StackOverflow:
       https://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
    """
    directories = [os.path.join(path, d) for d in os.listdir(path)
                   if os.path.isdir(os.path.join(path, d))]

    labels = []
    file_names = []
    for d in directories:
        _, label = os.path.split(d)
        labels += [int(label) for f in os.listdir(d) if f.endswith('.png')]
        file_names += [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.png')]

    return file_names, labels


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
        image: 3D tensor of [height, width, 1] of type float32.
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
            [image, label],
            batch_size=batch_size,
            num_threads=num_pre_process_threads,
            capacity=capacity,
            min_after_dequeue=min_queue_examples
        )
    else:
        image_batch, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_pre_process_threads,
            capacity=capacity
        )

    # Display the training images in the visualizer.
    tf.summary.image('images', image_batch, max_outputs=10)

    return image_batch, label_batch


def _data_set_stats(folder):
    """Display some general information about the used data set.

    Number of elements in the data set.
    Histogram showing the distribution of the different classes.
    Example images for each class.

    Args:
        folder (str): 'Training' or 'Testing' for the TS data set.

    Returns:
        nothing
    """
    path = os.path.join(DATA_PATH, folder)
    file_names_list, labels_list = _gen_labeled_input_list(path)

    print('{} set consists of a total of {} images.'.format(folder, len(file_names_list)))

    # Extract unique folders (labels)
    labels = [int(os.path.split(os.path.dirname(f))[1]) for f in file_names_list]

    # Plot class distribution histogram.
    plt.hist(labels, 62)
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()

    unique_labels = np.unique(labels)
    unique_labels_count = np.bincount(labels)

    # Plot sample pictures for each class.
    plt.figure(figsize=(15, 15))
    for i, label in enumerate(unique_labels):
        image_id = np.random.choice(np.where(labels == label)[0])
        image = data.imread(file_names_list[image_id])
        resized_image = np.array(transform.resize(image, INPUT_SHAPE[:2], mode="constant"))
        plt.subplot(8, 8, i + 1)
        plt.axis("off")
        plt.title("Label {} ({})".format(label, unique_labels_count[i]))
        plt.imshow(resized_image)
        plt.subplots_adjust(wspace=0.3)

    plt.show()


if __name__ == '__main__':
    _data_set_stats('Training')
    _data_set_stats('Testing')
