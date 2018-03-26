"""Training the model.

Tested with Python 3.6+. The Code should work with 3.4+.
"""

import time
from datetime import datetime
import tensorflow as tf

import s_model


# General TensorFlow settings and setup.
tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(1234)
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/s_train',
                           """Directory where to write event logs and checkpoints.""")
tf.app.flags.DEFINE_integer('max_steps', 10,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('log_frequency', 1,
                            """How often (every x steps) to log results to the console.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():
    """Train the network for a number of steps."""

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Prepare the training data on CPU, to avoid a possible slowdown in case some operations
        # are performed on GPU.
        with tf.device('/cpu:0'):
            images, labels = s_model.inputs_train()

        # Build the logits (prediction) graph.
        logits = s_model.inference(images)

        # Calculate loss.
        loss = s_model.loss(logits, labels)

        # Build the training graph, that updates the model parameters after each batch.
        train_op = s_model.train(loss, global_step)

        # Logging hook
        class _LoggerHook(tf.train.SessionRunHook):
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
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    print('{}: step {}, loss = {:.4f}, {:.1f} examples/sec ({:.3f} sec/batch)'
                          .format(datetime.now(), self._step, loss_value, examples_per_sec,
                                  sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            # The frequency, in number of global steps, that the summaries are written to disk
            # using a default summary saver.
            save_summaries_steps=FLAGS.log_frequency,
            # The frequency, in number of global steps, that the global step/sec is logged.
            log_step_count_steps=250,
            hooks=[
                # Requests stop at a specified step.
                tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                # Monitors the loss tensor and stops training if loss is NaN.
                tf.train.NanTensorHook(loss),
                _LoggerHook()
            ],
            config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement,
                gpu_options=tf.GPUOptions(allow_growth=True)
            )
        ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


# noinspection PyUnusedLocal
def main(argv=None):
    """TensorFlow starting routine."""
    # Delete old training data.
    if tf.gfile.Exists(FLAGS.train_dir):
        print('Deleting old checkpoint data from: "{}".'.format(FLAGS.train_dir))
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    # Start training.
    train()


if __name__ == '__main__':
    tf.app.run()
