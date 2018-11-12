"""Testing environment for the `tf.data` modules."""

import os
import tensorflow as tf
import tensorflow.contrib as tfc

from python.params import BASE_PATH, FLAGS
from python.load_sample import load_sample
from python.labels import ctoi


train_txt = os.path.join(BASE_PATH, 'data/train.txt')
batch_size = 2
assert os.path.exists(train_txt) and os.path.isfile(train_txt)


def train_input_fn():
    # Experimental function.

    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_generator(generator,
                                                 (tf.float32, tf.int32, tf.int32, tf.string),
                                                 (tf.TensorShape([None, 80]), tf.TensorShape([]),
                                                  tf.TensorShape([None]), tf.TensorShape([])),
                                                 args=[])

        # dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.padded_batch(batch_size=batch_size,
                                       padded_shapes=([None, 80], [], [None], []),
                                       drop_remainder=True)

        # dataset = dataset.prefetch(batch_size * 32)
        dataset = dataset.prefetch(4)

        return dataset.make_one_shot_iterator().get_next()


def generator(*args):
    print('ARGS:', args)

    with open(train_txt) as f:
        lines = f.readlines()

        for line in lines:
            path, label = map(lambda s: s.strip(), line.split(':', 1))
            path = os.path.join(BASE_PATH, 'data/corpus', path)

            spectrogram, spectrogram_length = load_sample(path)

            label_encoded = [ctoi(c) for c in label]

            yield spectrogram, spectrogram_length, label_encoded, label


# L8ER: Test function, remove later on.
if __name__ == '__main__':
    next_element = train_input_fn()

    with tf.Session() as session:
        # for example in range(FLAGS.num_examples_train):
        for example in range(1):
            print('DEBUG:', session.run(next_element))

    print('The End.')
