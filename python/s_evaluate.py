"""Evaluate the trained speech model.

L8ER: Add accuracy table.
"""

import math
from datetime import datetime

import numpy as np
import tensorflow as tf

from s_params import FLAGS
import s_model


def eval_once(summary_writer, med_op, wer_op, summary_op):
    """Run the evaluation once over all test/eval inputs.

    Args:
        summary_writer: Summary writer.
        summary_op: Summary operator.

    Returns:
        Nothing.
    """
    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver = tf.train.Saver()

            # Restore from checkpoint.
            saver.restore(sess, checkpoint.model_checkpoint_path)
            # Extract global stop from checkpoint.
            _, global_step = checkpoint.model_checkpoint_path.split('/')[-1].split('-'[-1])
            global_step = str(global_step)
            print('Loaded global step:', global_step)
        else:
            print('No checkpoint file found.')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = []
        try:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(FLAGS.num_examples_test / FLAGS.batch_size))
            med_sum, wer_sum = 0.0, 0.0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0

            while step < num_iter and not coord.should_stop():
                med_batch, wer_batch = sess.run([med_op, wer_op])
                med_sum += np.sum(med_batch)
                wer_sum += np.sum(wer_batch)
                step += 1
                print('DEBUG:', med_batch, wer_batch, step, num_iter)

            # Compute error rates.
            print('DEBUG1:', type(med_sum), type(wer_sum), type(global_step))
            mean_med = med_sum / num_iter
            mean_wer = wer_sum / num_iter
            print('{}: mean_edit_distance = {:.3f}; word_error_rate = {:.3f}'
                  .format(datetime.now(), mean_med, mean_wer))
            print('DEBUG2:', mean_med, med_sum, mean_wer, wer_sum, global_step)

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='eval/loss', simple_value=avg_loss)
            summary.value.add(tag='eval/mean_edit_distance', simple_value=mean_med)
            summary.value.add(tag='eval/word_error_rate', simple_value=mean_wer)
            # summary_writer.add_summary(summary, str(global_step))
            summary_writer.add_summary(summary, 1)
        except Exception as e:
            print('EXCEPTION:', e, ', type:', type(e))
            coord.request_stop(e)

        print('Stopping...')
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Evaluate the speech model."""
    with tf.Graph().as_default() as g:
        # Get evaluation sequences and ground truth.
        sequences, seq_length, labels, originals = s_model.inputs()

        # Build a graph that computes the logits predictions from the inference model.
        logits = s_model.inference(sequences, seq_length)

        # Calculate error rates
        med_op, wer_op = s_model.decoding(logits, seq_length, labels, originals)

        # Build the summary operation based on the TF collection of summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        # L8ER: Add continuous evaluation loop.
        eval_once(summary_writer, med_op, wer_op, summary_op)


# noinspection PyUnusedLocal
def main(argv=None):
    """TensorFlow starting routine."""
    # Delete old evaluation data
    if tf.gfile.Exists(FLAGS.eval_dir):
        print('Deleting old evaluation data from: "{}".'.format(FLAGS.eval_dir))
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    main()
