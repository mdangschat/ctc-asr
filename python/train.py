"""
Train the ASR model.

Tested with Python 3.5, 3.6 and 3.7. No Python 2 compatibility is being provided.
"""

import time

import tensorflow as tf

from input_functions import train_input_fn, dev_input_fn
from python.model import CTCModel
from python.params import FLAGS, get_parameters
from python.util import storage


# noinspection PyUnusedLocal
def main(argv=None):
    """TensorFlow starting routine."""

    # Delete old training data if requested.
    storage.maybe_delete_checkpoints(FLAGS.train_dir, FLAGS.delete)

    # TODO: Are those folders still needed?
    # Delete old validation/evaluation data if requested.
    eval_dir = '{}_dev'.format(FLAGS.train_dir)
    storage.maybe_delete_checkpoints(eval_dir, FLAGS.delete)

    # Delete old test data if requested.
    test_dir = '{}_test'.format(FLAGS.train_dir)
    storage.maybe_delete_checkpoints(test_dir, FLAGS.delete)

    # Logging information about the run.
    print('TensorFlow-Version: {}; Tag-Version: {}; Branch: {}; Commit: {}\nParameters: {}'
          .format(tf.VERSION, storage.git_latest_tag(), storage.git_branch(),
                  storage.git_revision_hash(), get_parameters()))

    # Setup TensorFlow run configuration and hooks.
    config = tf.estimator.RunConfig(
        tf_random_seed=FLAGS.random_seed,
        model_dir=FLAGS.train_dir,
        session_config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement,
            gpu_options=tf.GPUOptions(allow_growth=FLAGS.allow_vram_growth)
        )
    )

    model = CTCModel()

    # Construct the estimator that embodies the model.
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=FLAGS.train_dir,
        config=config
    )

    # Train the model.
    estimator.train(input_fn=train_input_fn, hooks=None, steps=None, max_steps=None)

    # Evaluate the trained model.
    evaluation_result = estimator.evaluate(input_fn=dev_input_fn, hooks=None, steps=None)
    tf.logging.info('Evaluation result: {}'.format(evaluation_result))

    # TODO: Removed for now. Complete training first.
    # prediction_result = estimator.predict(input_fn=pred_input_fn, predict_keys=[''])
    #
    # tf.logging.info('Completed all epochs.')


if __name__ == '__main__':
    # General TensorFlow setup.
    tf.logging.set_verbosity(tf.logging.INFO)
    random_seed = FLAGS.random_seed if FLAGS.random_seed != 0 else int(time.time())
    tf.set_random_seed(random_seed)

    # Run training.
    tf.app.run()
