"""Transcribe a given audio file."""

import os

import tensorflow as tf

from asr.params import FLAGS, TF_FLOAT, BASE_PATH
from asr.load_sample import load_sample, NUM_FEATURES
# WarpCTC crashes during evaluation. Even if it's only imported and not actually being used.
if FLAGS.use_warp_ctc:
    FLAGS.use_warp_ctc = False
    import asr.model as model
else:
    import asr.model as model


# File to transcribe.
# WAV_PATHS = ['../datasets/speech_data/timit/TIMIT/TRAIN/DR4/FALR0/SA1.WAV']
WAV_PATHS = ['/tmp/examples/audio1_16.wav',
             '/tmp/examples/donald_mono_16.wav',
             os.path.join(BASE_PATH, '../datasets/youdontunderstandme.wav')]


def transcribe_once(logits_op, decoded_op, plaintext_op, feed_dict):
    """Restore model from latest checkpoint and run the inference for the provided `sequence`.

    Args:
        logits_op (tf.Tensor):
            Logits operator.
        decoded_op (tf.Tensor):
            Decoded operator.
        plaintext_op (tf.Tensor):
            Plaintext operator.
        feed_dict (dict):
            Session run feed dictionary.

    Returns:
        Nothing.
    """
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

            if not coord.should_stop():
                logits, decoded, plaintext = sess.run([logits_op, decoded_op, plaintext_op],
                                                      feed_dict=feed_dict)

                print('Transcriptions {}:\n{}'.format(plaintext.shape, plaintext))

        except Exception as e:
            print('EXCEPTION:', e, ', type:', type(e))
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=120)


def transcribe(wav_file):
    """Load an audio file and prepare the TensorFlow graph for inference.

    Args:
        wav_file (str): Path to WAV file.

    Returns:
        Nothing.
    """
    assert os.path.isfile(wav_file)

    with tf.Graph().as_default():
        # Get evaluation sequences and ground truth.
        with tf.device('/cpu:0'):
            # Load audio file into tensor.
            sequence, seq_length = load_sample(wav_file)

            sequence = [sequence] * FLAGS.batch_size
            sequence_ph = tf.placeholder(dtype=TF_FLOAT,
                                         shape=[FLAGS.batch_size, None, NUM_FEATURES])

            seq_length = [seq_length] * FLAGS.batch_size
            seq_length_ph = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, 1])

            feed_dict = {
                sequence_ph: sequence,
                seq_length_ph: seq_length
            }

        # Build a graph that computes the logits predictions from the inference model.
        logits_op, seq_length = model.inference(sequence_ph, seq_length_ph, training=False)

        decoded_op, plaintext_op, _ = model.decode(logits_op, seq_length, originals=None)

        transcribe_once(logits_op, decoded_op, plaintext_op, feed_dict)


# noinspection PyUnusedLocal
def main(argv=None):
    """TensorFlow starting routine."""
    for wav_path in WAV_PATHS:
        transcribe(wav_path)


if __name__ == '__main__':
    main()
