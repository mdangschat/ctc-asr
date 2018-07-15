"""Evaluate the trained asr model.

L8ER: Add accuracy table.
"""

from datetime import datetime
import numpy as np
import tensorflow as tf

from asr.params import FLAGS
from asr.util import storage

# Evaluation specific flags.
tf.flags.DEFINE_boolean('test', False,
                        "`True if evaluation should use the test set, `False` if it should use the "
                        "dev set.")
tf.flags.DEFINE_string('eval_dir', '',
                       ("If set, evaluation log data will be stored here, instead of the default "
                        "directory `f'{FLAGS.train_dir}_eval'."))

# WarpCTC crashes during evaluation. Even if it's only imported and not actually being used.
if FLAGS.use_warp_ctc:
    FLAGS.use_warp_ctc = False
    import asr.model as model
else:
    import asr.model as model


# Which dataset TXT file to use for evaluation. 'test' or 'dev'.
__EVALUATION_TARGET = 'test' if FLAGS.test else 'dev'


def evaluate_once(loss_op, mean_ed_op, wer_op, summary_op, summary_writer):
    """Run the evaluation once over all test inputs.

    Args:
        loss_op (tf.Tensor): CTC loss operator.
        mean_ed_op (tf.Tensor): Mean Edit Distance operator.
        wer_op (tf.Tensor): Word Error Rate operator.
        summary_op (tf.Tensor): Summary merge operator.
        summary_writer (tf.FileWriter): Summary writer.

    Returns:
        Nothing.
    """
    # Session configuration.
    session_config = tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=tf.GPUOptions(allow_growth=FLAGS.allow_vram_growth)
    )

    if __EVALUATION_TARGET == 'test':
        num_target_samples = FLAGS.num_examples_test
    elif __EVALUATION_TARGET == 'dev':
        num_target_samples = FLAGS.num_examples_dev
    else:
        raise ValueError('Invalid evaluation target: "{}"'.format(__EVALUATION_TARGET))

    with tf.Session(config=session_config) as sess:
        checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            # Initialize network.
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            # Restore from checkpoint.
            saver = tf.train.Saver()
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

            num_iter = num_target_samples // FLAGS.batch_size
            loss_sum, mean_ed_sum, wer_sum = 0., 0., 0.
            step = 0
            summary_result = None

            while step < num_iter and not coord.should_stop():
                step += 1

                try:
                    summary_result, loss_batch, mean_ed_batch, wer_batch = sess.run([
                        summary_op, loss_op, mean_ed_op, wer_op
                    ])

                    loss_sum += np.sum(loss_batch)
                    mean_ed_sum += mean_ed_batch
                    wer_sum += wer_batch

                    print('{:%Y-%m-%d %H:%M:%S}: Step {:,d} of {:,d}; Results: loss={:7.3f}; '
                          'mean_edit_distance={:5.3f}; WER={:5.3f}'
                          .format(datetime.now(), step, num_iter, loss_batch,
                                  mean_ed_batch, wer_batch))

                except tf.errors.OutOfRangeError:
                    print('WARN: Due to not allowing for smaller final batches, '
                          '{} batches have not been evaluated.'.format(num_iter - step))
                    break

            # Compute error rates.
            avg_loss = loss_sum / step
            mean_ed = mean_ed_sum / step
            wer = wer_sum / step

            print('{:%Y-%m-%d %H:%M:%S}: Summarizing averages: '
                  'loss={:.3f}; mean_edit_distance={:.3f}; WER={:.3f}'
                  .format(datetime.now(), avg_loss, mean_ed, wer))

            summary = tf.Summary()
            summary.ParseFromString(summary_result)

            summary.value.add(tag='loss/ctc_loss', simple_value=avg_loss)
            summary.value.add(tag='loss/mean_edit_distance', simple_value=mean_ed)
            summary.value.add(tag='loss/word_error_rate', simple_value=wer)

            summary_writer.add_summary(summary, str(global_step))

        except Exception as e:
            print('EXCEPTION:', e, ', type:', type(e))
            coord.request_stop(e)

        print('Stopping evaluation...')
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=120)


def evaluate(eval_dir):
    """Evaluate the asr model.

    Args:
        eval_dir (str): Path where to store evaluation summaries.

    Returns:
        Nothing.
    """
    with tf.Graph().as_default() as graph:
        # Get evaluation sequences and ground truth.
        with tf.device('/cpu:0'):
            inputs = model.inputs(target=__EVALUATION_TARGET)
            sequences, seq_length, labels, label_length, originals = inputs

        # Build a graph that computes the logits predictions from the inference model.
        logits, seq_length = model.inference(sequences, seq_length, training=False)

        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            # Calculate error rates
            loss_op = model.loss(logits, seq_length, labels, label_length)

            decoded, plaintext, plaintext_summary = model.decode(logits, seq_length, originals)
            tf.summary.text('decoded_text', plaintext_summary[:, : FLAGS.num_samples_to_report])

            # Error metrics for decoded text.
            _, mean_ed_op, _, wer_op = model.decoded_error_rates(labels, originals, decoded,
                                                                 plaintext)

        # Build the summary operation based on the TF collection of summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(eval_dir, graph, flush_secs=10)

        # Maybe: Add continuous evaluation loop.
        evaluate_once(loss_op, mean_ed_op, wer_op, summary_op, summary_writer)


# noinspection PyUnusedLocal
def main(argv=None):
    """TensorFlow starting routine."""

    # Determine evaluation log directory.
    eval_dir = FLAGS.eval_dir if len(FLAGS.eval_dir) > 0 else '{}_{}'\
        .format(FLAGS.train_dir, __EVALUATION_TARGET)

    # Delete old evaluation data if requested.
    storage.maybe_delete_checkpoints(eval_dir, FLAGS.delete)

    evaluate(eval_dir)


if __name__ == '__main__':
    main()
