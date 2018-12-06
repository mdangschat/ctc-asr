"""
Validate the TensorFlow installation and availability of GFU support.
"""

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def test_environment():
    """
    Print TensorFlow installation information (GPU focused).

    Returns:
        Nothing.
    """
    print('TensorFlow version:', tf.VERSION)
    print('GPU device name:', tf.test.gpu_device_name())
    print('is GPU available:', tf.test.is_gpu_available())
    print('is build with CUDA:', tf.test.is_built_with_cuda())


if __name__ == '__main__':
    test_environment()
