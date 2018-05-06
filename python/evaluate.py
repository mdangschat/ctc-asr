"""Evaluate the trained speech model.

L8ER: Add accuracy table.
"""

import math
from datetime import datetime

import numpy as np
import tensorflow as tf

from python.params import FLAGS
import python.model as model


def eval_once(summary_writer, loss_op, mean_ed_op, wer_op, summary_op):
    """Run the evaluation once over all test inputs.
    TODO Documentation

    Args:
        summary_writer (): Summary writer.
        loss_op ():
        mean_ed_op ():
        wer_op ():
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
            loss_sum, mean_ed_sum, wer_sum = 0., 0., 0.
            step = 0

            while step < num_iter and not coord.should_stop():
                loss_batch, mean_ed_batch, wer_batch = sess.run([loss_op, mean_ed_op, wer_op])

                loss_sum += np.sum(loss_batch)
                mean_ed_sum += mean_ed_batch
                wer_sum += wer_batch
                step += 1
                print('{}: Batch {:5,d} results: loss={:.3f}; mean_edit_distance={:.3f}; WER={:.3f}'
                      .format(datetime.now(), step, loss_batch, mean_ed_batch, wer_batch))

            # Compute error rates.
            avg_loss = loss_sum / num_iter
            mean_ed = mean_ed_sum / num_iter
            wer = wer_sum / num_iter

            print('Summarizing averages:')
            print('{}: loss={:.3f}; mean_edit_distance={:.3f}; WER={:.3f}'
                  .format(datetime.now(), avg_loss, mean_ed, wer))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))

            summary.value.add(tag='loss/ctc_loss', simple_value=avg_loss)
            summary.value.add(tag='loss/mean_edit_distance', simple_value=mean_ed)
            # summary.value.add(tag='loss/edit_distances', histo=eds)
            summary.value.add(tag='loss/word_error_rate', simple_value=wer)
            # summary.value.add(tag='loss/word_error_rates', histo=wers)

            summary_writer.add_summary(summary, str(global_step))

        except Exception as e:
            print('EXCEPTION:', e, ', type:', type(e))
            coord.request_stop(e)

        print('Stopping...')
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate(eval_dir):
    """Evaluate the speech model.

    Args:
        eval_dir (str): Path where to store evaluation summaries.

    Returns:
        Nothing.
    """
    with tf.Graph().as_default() as g:
        # Get evaluation sequences and ground truth.
        sequences, seq_length, labels, originals = model.inputs()

        # Build a graph that computes the logits predictions from the inference model.
        logits = model.inference(sequences, seq_length)

        with tf.name_scope('loss'):
            # Calculate error rates
            loss_op = model.loss(logits, labels, seq_length)
            decoded, plaintext, plaintext_summary = model.decode(logits, seq_length, originals)
            tf.summary.text('decoded_text', plaintext_summary[:, : FLAGS.num_samples_to_report])

            # Error metrics for decoded text.
            eds, mean_ed_op, wers, wer_op = model.decoded_error_rates(labels, originals, decoded,
                                                                      plaintext)

            # tf.summary.histogram('edit_distances', eds)
            # tf.summary.histogram('word_error_rates', wers)

            # Build the summary operation based on the TF collection of summaries.
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(eval_dir, g)

            # L8ER: Add continuous evaluation loop.
            eval_once(summary_writer, loss_op, mean_ed_op, wer_op, summary_op)


# noinspection PyUnusedLocal
def main(argv=None):
    """TensorFlow starting routine."""

    # Determine evaluation log directory.
    eval_dir = FLAGS.eval_dir if len(FLAGS.eval_dir) > 0 else '{}_eval'.format(FLAGS.train_dir)

    # Delete old evaluation data if requested.
    if tf.gfile.Exists(eval_dir) and FLAGS.delete:
        print('Deleting old evaluation data from: {}.'.format(eval_dir))
        tf.gfile.DeleteRecursively(eval_dir)
    elif tf.gfile.Exists(eval_dir) and not FLAGS.delete:
        print('Resuming evaluation in: {}'.format(eval_dir))
    else:
        print('Starting a new evaluation in: {}'.format(eval_dir))
        tf.gfile.MakeDirs(eval_dir)

    evaluate(eval_dir)


if __name__ == '__main__':
    main()
