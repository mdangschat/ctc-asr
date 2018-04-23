"""Train the speech model.

Tested with Python 3.5 and 3.6.
Note: No Python 2 compatibility is provided.
"""

import tensorflow as tf

from params import FLAGS, get_parameters
from utils import get_git_branch, get_git_revision_hash, LoggerHook, TraceHook
import model


# General TensorFlow settings and setup.
tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(FLAGS.random_seed)


def train():
    """Train the network for a number of steps."""
    print('Version: {} Branch: {}'.format(get_git_revision_hash(), get_git_branch()))
    print('Parameters: ', get_parameters())

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Prepare the training data on CPU, to avoid a possible slowdown in case some operations
        # are performed on GPU.
        with tf.device('/cpu:0'):
            sequences, seq_length, labels, originals = model.inputs_train()

        # Build the logits (prediction) graph.
        logits = model.inference(sequences, seq_length)

        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            # Calculate loss/cost.
            loss = model.loss(logits, labels, seq_length)
            tf.summary.scalar('ctc_loss', loss)

            # Decode.
            decoded, plaintext, plaintext_summary = model.decode(logits, seq_length, originals)
            tf.summary.text('decoded_text', plaintext_summary[:, : FLAGS.num_samples_to_report])

            # Error metrics for decoded text.
            eds, mean_ed, wers, wer = model.decoded_error_rates(labels, originals, decoded,
                                                                plaintext)
            tf.summary.histogram('edit_distances', eds)
            tf.summary.scalar('mean_edit_distance', mean_ed)
            tf.summary.histogram('word_error_rates', wers)
            tf.summary.scalar('word_error_rate', wer)

        # Build the training graph, that updates the model parameters after each batch.
        train_op = model.train(loss, global_step)

        # Session configuration.
        session_config = tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement,
            gpu_options=tf.GPUOptions(allow_growth=FLAGS.allow_vram_growth)
        )

        # Summary hook.
        summary_op = tf.summary.merge_all()
        file_writer = tf.summary.FileWriterCache.get(FLAGS.train_dir)
        summary_saver_hook = tf.train.SummarySaverHook(save_steps=FLAGS.log_frequency,
                                                       summary_writer=file_writer,
                                                       summary_op=summary_op)

        # Session hooks.
        session_hooks = [
                # Requests stop at a specified step.
                tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                # Monitors the loss tensor and stops training if loss is NaN.
                tf.train.NanTensorHook(loss),
                # Summary saver hook.
                summary_saver_hook,
                # Monitor hook for TensorBoard to trace compute time, memory usage, and more.
                # Deactivated `TraceHook`, because it's computational intensive.
                # TraceHook(file_writer, FLAGS.log_frequency * 5),
                # LoggingHook.
                LoggerHook(loss)
            ]

        # The MonitoredTrainingSession takes care of session initialization, session resumption,
        # creating checkpoints, and some basic error handling.
        session = tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            save_checkpoint_steps=FLAGS.log_frequency * 5,
            # The frequency, in number of global steps, that the summaries are written to disk
            # using a default summary saver.
            save_summaries_steps=FLAGS.log_frequency,
            # The frequency, in number of global steps, that the global step/sec is logged.
            log_step_count_steps=FLAGS.log_frequency * 10,
            # Attach hooks to session.
            hooks=session_hooks,
            # Number of seconds given to threads to stop after close() has been called.
            stop_grace_period_secs=10,
            # Attach session config.
            config=session_config
        )

        with session:
            while not session.should_stop():
                try:
                    session.run([train_op])

                except tf.errors.OutOfRangeError:
                    print('All batches fed. Stopping.')
                    break


# noinspection PyUnusedLocal
def main(argv=None):
    """TensorFlow starting routine."""

    # Delete old training data if requested.
    if tf.gfile.Exists(FLAGS.train_dir):
        if FLAGS.delete:
            print('Deleting old checkpoint data from: {}.'.format(FLAGS.train_dir))
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
    else:
        print('Resuming training from: {}'.format(FLAGS.train_dir))

    tf.gfile.MakeDirs(FLAGS.train_dir)

    # Start training.
    train()


if __name__ == '__main__':
    tf.app.run()
