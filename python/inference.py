"""Transcribe a given audio file."""

import os
import math
from datetime import datetime

import numpy as np
import tensorflow as tf

from python.params import FLAGS
from python.loader.load_sample import load_sample
# WarpCTC crashes during evaluation. Even if it's only imported and not actually being used.
if FLAGS.use_warp_ctc:
    FLAGS.use_warp_ctc = False
    import python.model as model
else:
    import python.model as model


# File to transcribe.
WAV_FILE = '/home/marc/workspace/datasets/speech_data/timit/TIMIT/TRAIN/DR4/FALR0/SA1.WAV'


def transcribe_once(logits_op, decoded_op, plaintext_op):
    # TODO Document

    # Session configuration.
    session_config = tf.ConfigProto(
        log_device_placement=False,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )

    with tf.Session(config=session_config) as sess:
        checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver = tf.train.Saver()

            # Restore from checkpoint.
            saver.restore(sess, checkpoint.model_checkpoint_path)
            # Extract global stop from checkpoint.
            global_step = checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1]
            global_step = str(global_step)
            print('Loaded global step: {}, from checkpoint: {}'
                  .format(global_step, FLAGS.train_dir))
        else:
            print('No checkpoint file found.')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = []
        try:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = 1

            if not coord.should_stop():
                logits, decoded, plaintext = sess.run([logits_op, decoded_op, plaintext_op])

                print('logits:', logits.shape, logits)
                print('decoded:', decoded.shape, decoded)
                print('plaintext:', plaintext.shape, plaintext)

        except Exception as e:
            print('EXCEPTION:', e, ', type:', type(e))
            coord.request_stop(e)

        print('Stopping...')
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=120)


def transcribe():
    # TODO Document
    assert os.path.isfile(WAV_FILE)

    with tf.Graph().as_default() as graph:
        # Get evaluation sequences and ground truth.
        with tf.device('/cpu:0'):
            sequences, _ = load_sample(WAV_FILE)

        # Build a graph that computes the logits predictions from the inference model.
        logits, seq_length = model.inference(sequences, training=False)

        decoded, plaintext, _ = model.decode(logits, seq_length, originals=None)

        transcribe_once(logits, decoded, plaintext)


# noinspection PyUnusedLocal
def main(argv=None):
    """TensorFlow starting routine."""
    transcribe()


if __name__ == '__main__':
    main()
