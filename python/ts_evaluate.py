"""Evaluate the trained TS model.

Add accuracy table.
"""

import math
from datetime import datetime

import numpy as np
import tensorflow as tf

from traffic_signs import ts_input, ts_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/ts_eval',
                           """Directory where to write the evaluation logs into.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/ts_train',
                           """Points to the directory in which the training checkpoints are 
                           stored.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run the evaluation once over all test/eval inputs.

    Args:
        saver: Saver.
        summary_writer: Summary writer.
        top_k_op: Top K operator.
        summary_op: Summary operator.

    Returns:
        Nothing.
    """
    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            # Restore from checkpoint.
            saver.restore(sess, checkpoint.model_checkpoint_path)
            # Extract global stop from checkpoint.
            _, global_step = checkpoint.model_checkpoint_path.split('/')[-1].split('-'[-1])
            global_step = str(global_step)
            print('global_stop:', global_step)
        else:
            print('No checkpoint file found.')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = []
        try:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(ts_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('{}: precision @ 1 = {:.3f}'.format(datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, str(global_step[1]))
        except Exception as e:
            print('EXCEPTION:', e, ', type:', type(e))
            coord.request_stop(e)

        print('Stopping...')
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Evaluate TS."""
    # L8ER: Add top k accuracy support.
    top_k = 1
    with tf.Graph().as_default() as g:
        # Get images and labels.
        images, labels = ts_model.inputs(True)

        # Build a graph that computes the logits predictions from the inference model.
        logits = ts_model.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, top_k)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(ts_model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        # L8ER: Add continuous evaluation loop.
        eval_once(saver, summary_writer, top_k_op, summary_op)


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
