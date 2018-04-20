"""Evaluate the trained speech model.

L8ER: Add accuracy table.
"""

import math
from datetime import datetime

import numpy as np
import tensorflow as tf

from params import FLAGS
import model


def eval_once(summary_writer, loss_op, summary_op):
    """Run the evaluation once over all test inputs.
    TODO Documentation

    Args:
        summary_writer (): Summary writer.
        loss_op ():
        summary_op (): Summary operator.

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
            global_step = checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1]
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
            loss_sum = 0.0
            step = 0

            while step < num_iter and not coord.should_stop():
                loss_batch = sess.run([loss_op])
                loss_batch = loss_batch[0]

                loss_sum += np.sum(loss_batch)
                step += 1
                print('{}: loss={:.3f}; step={}'.format(datetime.now(), loss_batch, step))

            # Compute error rates.
            avg_loss = loss_sum / num_iter

            print('Summarizing averages:')
            print('{}: loss = {:.3f}'.format(datetime.now(), avg_loss))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='eval/ctc_loss', simple_value=avg_loss)
            summary_writer.add_summary(summary, str(global_step))

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
        sequences, seq_length, labels, originals = model.inputs()

        # Build a graph that computes the logits predictions from the inference model.
        logits = model.inference(sequences, seq_length)

        with tf.name_scope('eval'):
            # Calculate error rates
            loss_op = model.loss(logits, labels, seq_length)
            decoded, plaintext, plaintext_summary = model.decode(logits, seq_length, originals)
            tf.summary.text('decoded_text', plaintext_summary[:, : FLAGS.num_samples_to_report])

            # Error metrics for decoded text.
            eds, mean_ed, wers, wer = model.decoded_error_rates(labels, originals, decoded,
                                                                plaintext)
            tf.summary.histogram('edit_distances', eds)
            tf.summary.scalar('mean_edit_distance', mean_ed)
            tf.summary.histogram('word_error_rates', wers)
            tf.summary.scalar('word_error_rate', wer)

            # Build the summary operation based on the TF collection of summaries.
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

            # L8ER: Add continuous evaluation loop.
            eval_once(summary_writer, loss_op, summary_op)


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
