"""
Evaluate a trained ASR model.
"""

import tensorflow as tf

from asr.input_functions import input_fn_generator
from asr.model import CTCModel
from asr.params import FLAGS, get_parameters
from asr.util import storage

# Evaluation specific flags.
tf.flags.DEFINE_boolean('dev', False,
                        "`True` if evaluation should use the dev set, `False` if it should use the"
                        " test set.")

# Which dataset TXT file to use for evaluation. 'test' or 'dev'.
__EVALUATION_TARGET = 'dev' if FLAGS.dev else 'test'


def main(_):
    """TensorFlow evaluation starting routine."""

    # Delete old model data if requested.
    storage.maybe_delete_checkpoints(FLAGS.train_dir, FLAGS.delete)

    # Logging information about the run.
    print('TensorFlow-Version: {}; Tag-Version: {}; Branch: {}; Commit: {}\nParameters: {}'
          .format(tf.VERSION, storage.git_latest_tag(), storage.git_branch(),
                  storage.git_revision_hash(), get_parameters()))

    # Setup TensorFlow run configuration and hooks.
    config = tf.estimator.RunConfig(
        model_dir=FLAGS.train_dir,
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

    # Evaluate the trained model.
    dev_input_fn = input_fn_generator(__EVALUATION_TARGET)
    evaluation_result = estimator.evaluate(input_fn=dev_input_fn, hooks=None)
    tf.logging.info('Evaluation results for this model: {}'.format(evaluation_result))


if __name__ == '__main__':
    # General TensorFlow setup.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Run training.
    tf.app.run()
