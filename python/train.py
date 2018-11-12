"""Train the asr model.

Tested with Python 3.5, 3.6 and 3.7.
No Python 2 compatibility is being provided.
"""

import time
import tensorflow as tf
import tensorflow.contrib as tfc

from python.params import FLAGS, get_parameters, TF_FLOAT
from python.util import storage
from python.model import model_fn
# from python.s_input import train_input_fn, eval_input_fn, pred_input_fn
# from python.util.hooks import LoggerHook
from python.dataset.corpus_input import train_input_fn


def hooks():
    # TODO Documentation
    # TODO Assemble hooks.

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

    # Session hooks.
    session_hooks = [
        # Monitors the loss tensor and stops training if loss is NaN.
        # tf.train.NanTensorHook(loss), TODO fix
        # Checkpoint saver hook.
        checkpoint_saver_hook,
        # Summary saver hook.
        summary_saver_hook,
        # Monitor hook for TensorBoard to trace compute time, memory usage, and more.
        # Deactivated `TraceHook`, because it's computational intensive.
        # TraceHook(file_writer, FLAGS.log_frequency * 5),
        # LoggingHook.
        # LoggerHook(loss)    # TODO fix
    ]

    return session_hooks


# noinspection PyUnusedLocal
def main(argv=None):
    """TensorFlow starting routine."""

    # Delete old training data if requested.
    storage.maybe_delete_checkpoints(FLAGS.train_dir, FLAGS.delete)

    # Delete old validation/evaluation data if requested.
    eval_dir = '{}_dev'.format(FLAGS.train_dir)
    storage.maybe_delete_checkpoints(eval_dir, FLAGS.delete)

    # Delete old test data if requested.
    test_dir = '{}_test'.format(FLAGS.train_dir)
    storage.maybe_delete_checkpoints(test_dir, FLAGS.delete)

    # Logging information about the run.
    tf.logging.info('TensorFlow-Version: {}; Tag-Version: {}; Branch: {}; Commit: {}\n'
                    'Parameters: {}'
                    .format(tf.VERSION, storage.git_latest_tag(), storage.git_branch(),
                            storage.git_revision_hash(), get_parameters()))

    # Setup TensorFlow run configuration and hooks.
    config = tf.estimator.RunConfig(
        session_config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement,
            gpu_options=tf.GPUOptions(allow_growth=FLAGS.allow_vram_growth)
        )
    )

    # Iterate until the complete dataset is consumed.
    steps = None

    # TODO Feature transform function.
    # spectrogram_fc = tf.feature_column.numeric_column(key='spectrogram',
    #                                                   shape=(None, 80),
    #                                                   dtype=TF_FLOAT)
    # spectrogram_length_fc = tf.feature_column.numeric_column(key='spectrogram_length',
    #                                                          shape=(1, ),
    #                                                          dtype=TF_FLOAT)
    spectrogram_fc = tfc.feature_column.sequence_numeric_column(key='spectrogram')
    spectrogram_length_fc = tfc.feature_column.sequence_numeric_column(key='spectrogram_length')

    tf.parse_example(features=tf.feature_column.make_parse_example_spec(spectrogram_fc))

    # Construct the estimator that embodies the model.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.train_dir,
        config=config,
        params={
            'feature_columns': [spectrogram_fc, spectrogram_length_fc]
        }
    )

    # Train the model.
    estimator.train(input_fn=train_input_fn, hooks=hooks(), steps=steps)

    # TODO: Removed for now. Complete training first.
    # # Evaluate the trained model.
    # evaluation_result = estimator.evaluate(input_fn=eval_input_fn, hooks=hooks, steps=None,
    #                                        checkpoint_path=eval_dir)
    # tf.logging.info('Evaluation result: {}'.format(evaluation_result))
    #
    # prediction_result = estimator.predict(input_fn=pred_input_fn, predict_keys=[''])
    #
    # tf.logging.info('Completed all epochs.')


if __name__ == '__main__':
    # General TensorFlow setup.
    tf.logging.set_verbosity(tf.logging.INFO)
    random_seed = FLAGS.random_seed if FLAGS.random_seed != 0 else int(time.time())
    tf.set_random_seed(FLAGS.random_seed)

    # Run training.
    tf.app.run()
