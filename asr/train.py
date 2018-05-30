"""Train the asr model.

Tested with Python 3.5 and 3.6.
Note: No Python 2 compatibility is being provided.
"""

import tensorflow as tf
from datetime import datetime

from asr.params import FLAGS, get_parameters
from asr.util import storage, tf_contrib
import asr.model as model


# General TensorFlow settings and setup.
tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(FLAGS.random_seed)


def train(shuffle):
    """Train the network for a number of steps. Uses SortaGrad to start training (first epoch)
     on short sequences, and increase their length throughout the first epoch. After that it
     uses shuffled inputs.
     This leads to a very low loss at the beginning of training, and also much faster computation
     times at the start of the first epoch.

    Args:
        shuffle (bool): Whether to use sorted inputs or shuffled inputs for training.
            See SortaGrad approach as documented in `Deep Speech 2`_.

    .. _`Deep Speech 2`:
        https://arxiv.org/abs/1512.02595
    """
    print('Version: {} Branch: {} Commit: {}'
          .format(storage.git_latest_tag(), storage.git_branch(), storage.git_revision_hash()))
    print('Parameters: ', get_parameters())
    print('SortaGrad shuffle active: ', shuffle)

    max_steps_epoch = FLAGS.num_examples_train // FLAGS.batch_size
    max_steps_total = max_steps_epoch * FLAGS.max_epochs

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Prepare the training data on CPU, to avoid a possible slowdown in case some operations
        # are performed on GPU.
        with tf.device('/cpu:0'):
            # Note that _ is `seq_length`, which is calculated in `inference()` for now.
            sequences, _, labels, label_length, originals = model.inputs_train(shuffle)

        # Build the logits (prediction) graph.
        logits, seq_length = model.inference(sequences)

        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            # Calculate loss/cost.
            loss = model.loss(logits, seq_length, labels, label_length)
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

        # Checkpoint saver hook.
        checkpoint_saver = tf.train.Saver(
            # Note: cuDNN RNNs do not support distributed saving of parameters.
            sharded=False,
            allow_empty=True,
            max_to_keep=50,
            save_relative_paths=True
        )

        checkpoint_saver_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir=FLAGS.train_dir,
            save_secs=None,
            save_steps=FLAGS.log_frequency * 10,
            saver=checkpoint_saver
        )

        # Summary hook.
        summary_op = tf.summary.merge_all()
        file_writer = tf.summary.FileWriterCache.get(FLAGS.train_dir)
        summary_saver_hook = tf.train.SummarySaverHook(save_steps=FLAGS.log_frequency,
                                                       summary_writer=file_writer,
                                                       summary_op=summary_op)

        # Stop after steps hook.
        last_step = max_steps_total if shuffle else max_steps_epoch
        stop_step_hook = tf.train.StopAtStepHook(last_step=last_step)

        # Session hooks.
        session_hooks = [
            # Requests stop at a specified step.
            stop_step_hook,
            # Monitors the loss tensor and stops training if loss is NaN.
            tf.train.NanTensorHook(loss),
            # Checkpoint saver hook.
            checkpoint_saver_hook,
            # Summary saver hook.
            summary_saver_hook,
            # Monitor hook for TensorBoard to trace compute time, memory usage, and more.
            # Deactivated `TraceHook`, because it's computational intensive.
            # TraceHook(file_writer, FLAGS.log_frequency * 5),
            # LoggingHook.
            tf_contrib.LoggerHook(loss)
        ]

        # The MonitoredTrainingSession takes care of session initialization, session resumption,
        # creating checkpoints, and some basic error handling.
        session = tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            save_checkpoint_steps=None,
            save_checkpoint_secs=None,
            # Don't use the sessions default summary saver.
            save_summaries_steps=None,
            save_summaries_secs=None,
            # The frequency, in number of global steps, that the global_step/sec is logged.
            log_step_count_steps=FLAGS.log_frequency * 10,
            # Set session scaffolding.
            scaffold=tf.train.Scaffold(saver=checkpoint_saver),
            # Attach hooks to session.
            hooks=session_hooks,
            # Number of seconds given to threads to stop after close() has been called.
            stop_grace_period_secs=30,
            # Attach session config.
            config=session_config
        )

        with session:
            current_global_step = tf.train.global_step(session, global_step)
            current_global_step += 1  # Offset accounts for TF counting from 0.

            if (shuffle and current_global_step >= max_steps_epoch) or \
               (not shuffle and current_global_step < max_steps_epoch):
                while not session.should_stop():
                    try:
                        _, current_global_step = session.run([train_op, global_step])

                    except tf.errors.OutOfRangeError:
                        print('{:%Y-%m-%d %H:%M:%S}: All batches fed. Stopping.'
                              .format(datetime.now()))
                        # If `run` isn't successful, global step isn't being updated.
                        current_global_step += 1
                        break

    # Switch to shuffle if the first epoch has finished. See SortaGrad.
    current_global_step += 1
    if not shuffle and current_global_step >= max_steps_epoch:
        train(True)


# noinspection PyUnusedLocal
def main(argv=None):
    """TensorFlow starting routine."""

    # Delete old training data if requested.
    if tf.gfile.Exists(FLAGS.train_dir) and FLAGS.delete:
        print('Deleting old checkpoint data from: {}.'.format(FLAGS.train_dir))
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    elif tf.gfile.Exists(FLAGS.train_dir) and not FLAGS.delete:
        print('Resuming training from: {}'.format(FLAGS.train_dir))
    else:
        print('Starting a new training run in: {}'.format(FLAGS.train_dir))
        tf.gfile.MakeDirs(FLAGS.train_dir)

    # Start training. `shuffle=False` indicates that the 1st epoch uses SortaGrad.
    train(shuffle=False)


if __name__ == '__main__':
    tf.app.run()
