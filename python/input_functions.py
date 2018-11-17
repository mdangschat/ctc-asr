"""
Routines to load a corpus and perform the necessary pre processing on the audio files and labels.
"""

import os

import tensorflow as tf

from python.labels import ctoi
from python.load_sample import load_sample
from python.params import BASE_PATH, BOUNDARIES, FLAGS


def train_input_fn():
    return _input_fn(FLAGS.train_csv, use_buckets=True, epochs=FLAGS.max_epochs)


def test_input_fn():
    return _input_fn(FLAGS.test_csv, use_buckets=False, epochs=1)


def dev_input_fn():
    # TODO: For testing, since dev.csv does not exist in correct format
    # return _input_fn(FLAGS.dev_csv, use_buckets=True, epochs=1)
    return test_input_fn()


def _input_fn(csv_path, use_buckets=True, epochs=2):
    # TODO: Documentation.
    # TODO: Debug defaults.

    def element_length_fn(_spectrogram, _spectrogram_length, _label_encoded, _label_plaintext):
        del _spectrogram
        del _label_encoded
        del _label_plaintext
        return _spectrogram_length

    assert os.path.exists(csv_path) and os.path.isfile(csv_path)

    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_generator(_input_generator,
                                                 (tf.float32, tf.int32, tf.int32, tf.string),
                                                 (tf.TensorShape([None, 80]), tf.TensorShape([]),
                                                  tf.TensorShape([None]), tf.TensorShape([])),
                                                 args=[csv_path])

        if use_buckets:
            dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
                element_length_func=element_length_fn,
                bucket_boundaries=BOUNDARIES,
                bucket_batch_sizes=[FLAGS.batch_size] * (len(BOUNDARIES) + 1),
                pad_to_bucket_boundary=False,
                no_padding=False))

        else:
            dataset = dataset.padded_batch(batch_size=FLAGS.batch_size,
                                           padded_shapes=([None, 80], [], [None], []),
                                           drop_remainder=True)

        # dataset.cache()
        dataset = dataset.prefetch(64)

        # Number of epochs.
        dataset = dataset.repeat(epochs)

        iterator = dataset.make_one_shot_iterator()
        spectrogram, spectrogram_length, label_encoded, label_plaintext = iterator.get_next()

        features = {
            'spectrogram': spectrogram,
            'spectrogram_length': spectrogram_length,
            'label_plaintext': label_plaintext
        }

        return features, label_encoded


def _input_generator(*args):
    # TODO: Documentation
    # TODO: Use CSV

    with open(args[0]) as f:
        lines = f.readlines()
        lines = lines[1:]  # Remove CSV header.

        for line in lines:
            path, label = map(lambda s: s.strip(), line.split(';', 1))
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
