"""
TODO: Documentation.
"""

import os
import tensorflow as tf

from python.params import BASE_PATH, FLAGS
from python.load_sample import load_sample
from python.labels import ctoi


def train_input_fn():
    return _input_fn(FLAGS.train_txt)


def test_input_fn():
    return _input_fn(FLAGS.test_txt)


def dev_input_fn():
    # TODO: For testing, since dev.txt does not exist in correct format
    # return _input_fn(FLAGS.dev_txt)
    return train_input_fn()


def _input_fn(txt_path):
    # TODO: Documentation.

    assert os.path.exists(txt_path) and os.path.isfile(txt_path)

    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_generator(input_generator,
                                                 (tf.float32, tf.int32, tf.int32, tf.string),
                                                 (tf.TensorShape([None, 80]), tf.TensorShape([]),
                                                  tf.TensorShape([None]), tf.TensorShape([])),
                                                 args=[txt_path])

        dataset = dataset.padded_batch(batch_size=FLAGS.batch_size,
                                       padded_shapes=([None, 80], [], [None], []),
                                       drop_remainder=True)

        dataset = dataset.prefetch(32)

        # TODO: Number of epochs.
        dataset = dataset.repeat(1)

        iterator = dataset.make_one_shot_iterator()
        spectrogram, spectrogram_length, label_encoded, label_plaintext = iterator.get_next()

        features = {
            'spectrogram': spectrogram,
            'spectrogram_length': spectrogram_length,
            'label_plaintext': label_plaintext
        }

        return features, label_encoded


def input_generator(*args):
    # TODO: Documentation

    with open(args[0]) as f:
        lines = f.readlines()

        for line in lines:
            path, label = map(lambda s: s.strip(), line.split(':', 1))
            path = os.path.join(BASE_PATH, 'data/corpus', path)

            spectrogram, spectrogram_length = load_sample(path)

            label_encoded = [ctoi(c) for c in label]

            yield spectrogram, spectrogram_length, label_encoded, label


# Create a dataset for testing purposes.
if __name__ == '__main__':
    next_element = train_input_fn()

    with tf.Session() as session:
        # for example in range(FLAGS.num_examples_train):
        for example in range(5):
            print('Dataset elements:', session.run(next_element))

    print('The End.')
