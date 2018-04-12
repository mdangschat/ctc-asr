"""Train the model.

Tested with Python 3.6.4. The Code should work with 3.4+.
Python 2.x compatibility isn't provided.
"""

import time
from datetime import datetime
import tensorflow as tf

from s_params import FLAGS, get_parameters
import s_model
from s_utils import get_git_branch, get_git_revision_hash


# General TensorFlow settings and setup.
tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(4711)


def train():
    """Train the network for a number of steps."""
    print('Version: {}; Branch: {}'.format(get_git_revision_hash(), get_git_branch()))
    print('Parameters: ', get_parameters())

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Prepare the training data on CPU, to avoid a possible slowdown in case some operations
        # are performed on GPU.
        with tf.device('/cpu:0'):
            sequences, seq_length, labels, originals = s_model.inputs_train()

        # Build the logits (prediction) graph.
        logits = s_model.inference(sequences, seq_length)

        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            # Calculate loss/cost.
            loss = s_model.loss(logits, labels, seq_length)

            # Decode.
            _, _ = s_model.decoding(logits, seq_length, labels, originals)

        # Build the training graph, that updates the model parameters after each batch.
        train_op = s_model.train(loss, global_step)

        # Logging hook
        class LoggerHook(tf.train.SessionRunHook):
            """Log loss and runtime."""

            def __init__(self):
                self._start_time = 0
                self._step = 0

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)    # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = duration / float(FLAGS.log_frequency)

                    print('{}: step {}, loss={:.4f}, {:.1f} examples/sec ({:.3f} sec/batch)'
                          .format(datetime.now(), self._step, loss_value,
                                  examples_per_sec, sec_per_batch))

        # Session configuration.
        session_config = tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )

        # Session hooks.
        session_hooks = [
                # Requests stop at a specified step.
                tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                # Monitors the loss tensor and stops training if loss is NaN.
                tf.train.NanTensorHook(loss),
                LoggerHook()
            ]

        # The MonitoredTrainingSession takes care of session initialization, session resumption,
        # creating checkpoints, and some basic error handling.
        session = tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            # The frequency, in number of global steps, that the summaries are written to disk
            # using a default summary saver.
            save_summaries_steps=FLAGS.log_frequency,
            # The frequency, in number of global steps, that the global step/sec is logged.
            log_step_count_steps=FLAGS.log_frequency * 5,
            hooks=session_hooks,
            config=session_config
        )

        with session:
            while not session.should_stop():
                try:
                    session.run(train_op)
                except tf.errors.OutOfRangeError:
                    print('All batches fed. Stopping.')
                    break


# noinspection PyUnusedLocal
def main(argv=None):
    """TensorFlow starting routine."""
    # Delete old training data.
    if tf.gfile.Exists(FLAGS.train_dir):
        print('Deleting old checkpoint data from: {}.'.format(FLAGS.train_dir))
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    # Start training.
    train()


if __name__ == '__main__':
    tf.app.run()
