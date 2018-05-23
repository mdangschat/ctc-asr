"""Transcribe a given audio file."""

import os

import tensorflow as tf

from python.params import FLAGS, TF_FLOAT
from python.loader.load_sample import load_sample, NUM_FEATURES
# WarpCTC crashes during evaluation. Even if it's only imported and not actually being used.
if FLAGS.use_warp_ctc:
    FLAGS.use_warp_ctc = False
    import python.model as model
else:
    import python.model as model


# File to transcribe.
WAV_FILE = '/home/marc/workspace/datasets/speech_data/timit/TIMIT/TRAIN/DR4/FALR0/SA1.WAV'


def transcribe_once(logits_op, decoded_op, plaintext_op, sequences, sequences_ph):
    """Restore model from latest checkpoint and run the inference for the provided `sequence`.

    Args:
        logits_op (tf.Tensor):
            Logits operator.
        decoded_op (tf.Tensor):
            Decoded operator.
        plaintext_op (tf.Tensor):
            Plaintext operator.
        sequences (List[np.ndarray]):
            Python list of 2D numpy arrays, each containing audio features.
        sequences_ph (tf.Tensor):
            Placeholder for the input sequences.

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
                                                      feed_dict={sequences_ph: sequences})

                print('Transcriptions {}:\n{}'.format(plaintext.shape, plaintext))

        except Exception as e:
            print('EXCEPTION:', e, ', type:', type(e))
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=120)


def transcribe():
    """Load an audio file and prepare the TensorFlow graph for inference.

    Returns:
        Nothing.
    """
    assert os.path.isfile(WAV_FILE)

    with tf.Graph().as_default():
        # Get evaluation sequences and ground truth.
        with tf.device('/cpu:0'):
            # Load audio file into tensor.
            sequences, _ = load_sample(WAV_FILE)
            sequences = [sequences] * FLAGS.batch_size
            sequences_ph = tf.placeholder(dtype=TF_FLOAT,
                                          shape=[FLAGS.batch_size, None, NUM_FEATURES])

        # Build a graph that computes the logits predictions from the inference model.
        logits_op, seq_length = model.inference(sequences_ph, training=False)

        decoded_op, plaintext_op, _ = model.decode(logits_op, seq_length, originals=None)

        transcribe_once(logits_op, decoded_op, plaintext_op, sequences, sequences_ph)


# noinspection PyUnusedLocal
def main(argv=None):
    """TensorFlow starting routine."""
    transcribe()


if __name__ == '__main__':
    main()
