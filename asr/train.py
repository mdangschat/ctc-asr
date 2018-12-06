"""
Train the ASR model.

Tested with Python 3.5, 3.6 and 3.7. No Python 2 compatibility is being provided.
"""

import time

import tensorflow as tf

from asr.input_functions import input_fn_generator
from asr.model import CTCModel
from asr.params import FLAGS, get_parameters
from asr.util import storage

RANDOM_SEED = FLAGS.random_seed if FLAGS.random_seed != 0 else int(time.time())


def main(_):
    """TensorFlow starting routine."""

    # Delete old model data if requested.
    storage.maybe_delete_checkpoints(FLAGS.train_dir, FLAGS.delete)

    # Logging information about the run.
    print('TensorFlow-Version: {}; Tag-Version: {}; Branch: {}; Commit: {}\nParameters: {}'
          .format(tf.VERSION, storage.git_latest_tag(), storage.git_branch(),
                  storage.git_revision_hash(), get_parameters()))

    # Setup TensorFlow run configuration and hooks.
    config = tf.estimator.RunConfig(
        model_dir=FLAGS.train_dir,
        tf_random_seed=RANDOM_SEED,
        save_summary_steps=FLAGS.log_frequency,
        session_config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement,
            gpu_options=tf.GPUOptions(allow_growth=FLAGS.allow_vram_growth)
        ),
        keep_checkpoint_max=5,
        log_step_count_steps=FLAGS.log_frequency,
        train_distribute=None
    )

    model = CTCModel()

    # Construct the estimator that embodies the model.
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=FLAGS.train_dir,
        config=config
    )

    # Train the model.
    curriculum_train_input_fn = input_fn_generator('train_batch')
    estimator.train(input_fn=curriculum_train_input_fn, hooks=None)

    # Evaluate the trained model.
    dev_input_fn = input_fn_generator('dev')
    evaluation_result = estimator.evaluate(input_fn=dev_input_fn, hooks=None)
    tf.logging.info('Evaluation results of epoch {}: {}'.format(1, evaluation_result))

    # Train the model and evaluate after each epoch.
    for epoch in range(2, FLAGS.max_epochs + 1):
        # Train the model.
        train_input_fn = input_fn_generator('train_bucket')
        estimator.train(input_fn=train_input_fn, hooks=None)

        # TODO: Possible replacement for evaluate every epoch:
        # https://www.tensorflow.org/api_docs/python/tf/contrib/estimator/InMemoryEvaluatorHook

        # Evaluate the trained model.
        dev_input_fn = input_fn_generator('dev')
        evaluation_result = estimator.evaluate(input_fn=dev_input_fn, hooks=None)
        tf.logging.info('Evaluation results of epoch {}: {}'.format(epoch, evaluation_result))


if __name__ == '__main__':
    # General TensorFlow setup.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.set_random_seed(RANDOM_SEED)

    # Run training.
    tf.app.run()
