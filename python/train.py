"""Train the asr model.

Tested with Python 3.5 and 3.6.
Note: No Python 2 compatibility is being provided.
"""

import time

import tensorflow as tf
from datetime import datetime

from python.params import FLAGS, get_parameters
from python.util import storage
import python.model as model
from python.evaluate import evaluate

from python.util.hooks import GPUStatisticsHook, LoggerHook


# General TensorFlow settings and setup.
tf.logging.set_verbosity(tf.logging.INFO)
__random_seed = FLAGS.random_seed if FLAGS.random_seed != 0 else int(time.time())
tf.set_random_seed(FLAGS.random_seed)

__STEPS_EPOCH = (FLAGS.num_examples_train // FLAGS.batch_size) - 1
__MAX_STEPS = __STEPS_EPOCH * FLAGS.max_epochs


def train(epoch):
    """Train the network for a number of steps. Uses SortaGrad to start training (first epoch)
     on short sequences, and increase their length throughout the first epoch. After that it
     uses shuffled inputs.
     This leads to a very low loss at the beginning of training, and also much faster computation
     times at the start of the first epoch.

    Args:
        epoch (int): Whether to use sorted inputs (`epoch=0`) or shuffled inputs for training.
            See SortaGrad approach as documented in `Deep Speech 2`_.

    Returns:
        bool: True if all training examples of the epoch are completed, else False.

    .. _`Deep Speech 2`:
        https://arxiv.org/abs/1512.02595
    """
    current_global_step = -1
    print('Starting epoch {}.'.format(epoch))

    with tf.Graph().as_default():
        # Prepare the training data on CPU, to avoid a possible slowdown in case some operations
        # are performed on GPU.
        with tf.device('/cpu:0'):
            global_step = tf.train.get_or_create_global_step()
            sequences, seq_length, labels, label_length, originals = model.inputs_train(epoch > 0)

        # Build the logits (prediction) graph.
        logits, seq_length = model.inference(sequences, seq_length, training=True)

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
            max_to_keep=20,
            save_relative_paths=True
        )

        checkpoint_saver_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir=FLAGS.train_dir,
            save_secs=None,
            save_steps=FLAGS.log_frequency * 40,
            saver=checkpoint_saver
        )

        # Summary hook.
        summary_op = tf.summary.merge_all()
        file_writer = tf.summary.FileWriterCache.get(FLAGS.train_dir)
        summary_saver_hook = tf.train.SummarySaverHook(save_steps=FLAGS.log_frequency,
                                                       summary_writer=file_writer,
                                                       summary_op=summary_op)

        # GPU statistics hook.
        gpu_stats_hook = GPUStatisticsHook(
            log_every_n_steps=FLAGS.log_frequency,
            query_every_n_steps=FLAGS.gpu_hook_query_frequency,
            average_n=FLAGS.gpu_hook_average_queries,
            stats=['mem_util', 'gpu_util'],
            summary_writer=file_writer,
            suppress_stdout=False,
            group_tag='gpus'
        )

        # Stop after steps hook.
        last_step = (epoch + 1) * __STEPS_EPOCH
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
            # GPU statistics hook.
            gpu_stats_hook,
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
            # Don't use the sessions default checkpoint saver.
            save_checkpoint_steps=None,
            save_checkpoint_secs=None,
            # Don't use the sessions default summary saver.
            save_summaries_steps=None,
            save_summaries_secs=None,
            # The frequency, in number of global steps, that the global_step/sec is logged.
            log_step_count_steps=FLAGS.log_frequency * 20,
            # Set session scaffolding.
            scaffold=tf.train.Scaffold(saver=checkpoint_saver),
            # Attach hooks to session.
            hooks=session_hooks,
            # Number of seconds given to threads to stop after close() has been called.
            stop_grace_period_secs=60,
            # Attach session config.
            config=session_config
        )

        with session:
            while not session.should_stop():
                try:
                    _, current_global_step = session.run([train_op, global_step])

                except tf.errors.OutOfRangeError:
                    print('{:%Y-%m-%d %H:%M:%S}: All batches of epoch fed.'
                          .format(datetime.now()))
                    break

    return current_global_step


# noinspection PyUnusedLocal
def main(argv=None):
    """TensorFlow starting routine."""

    # Delete old training data if requested.
    storage.maybe_delete_checkpoints(FLAGS.train_dir, FLAGS.delete)

    # Delete old evaluation data if requested.
    eval_dir = '{}_dev'.format(FLAGS.train_dir)
    storage.maybe_delete_checkpoints(eval_dir, FLAGS.delete)

    # Delete old test data if requested.
    storage.maybe_delete_checkpoints('{}_test'.format(FLAGS.train_dir), FLAGS.delete)

    # Logging information's about the run.
    print('Version: {} Branch: {} Commit: {}'
          .format(storage.git_latest_tag(), storage.git_branch(), storage.git_revision_hash()))
    print('Parameters: ', get_parameters())

    # Calculate global_step and epoch.
    current_global_step = storage.maybe_read_global_step(FLAGS.train_dir)
    epoch = 0 if current_global_step < __STEPS_EPOCH else current_global_step // __STEPS_EPOCH
    if current_global_step > 0 and epoch == 0:
        print("""WARN: Resumed before completing the first epoch.
              This can lead to compromised learning due to the status of the SortaGrad input queue
              order isn't being stored in the checkpoint.""")

    # Main training loop.
    while current_global_step < __MAX_STEPS:
        print('Starting training at epoch {}, global_step {}.'.format(epoch, current_global_step))
        # Start training. `epoch=0` indicates that the 1st epoch uses SortaGrad.
        current_global_step = train(epoch)

        # Validate results after each epoch.
        if current_global_step > 1:
            print('Starting evaluation after the {} epoch.'.format(epoch))
            evaluate(eval_dir)

        epoch += 1

    print('Completed all epochs.')


if __name__ == '__main__':
    tf.app.run()
