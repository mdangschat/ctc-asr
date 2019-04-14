"""
Contains the ASR system's model definition.
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc

from asr.params import FLAGS, TF_FLOAT
from asr.util import tf_contrib, metrics
from asr.util.hooks import GPUStatisticsHook


class CTCModel:
    """Container class for the ASR system's TensorFlow model."""

    def __init__(self):
        # Initialize attributes.
        self.loss_op = None
        self.train_op = None
        self.hooks = None

    def model_fn(self, features, labels, mode):
        """Create model graph and return a configured estimator.

        Model function that constructs the model's graph and returns the configured
        `tf.estimator.EstimatorSpec`.

        This method sets the `self.loss_op` and `self.train_op` variables.

        Args:
            features (tf.Tensor or Dict[tf.Tensor]):
                This is the first item returned from the `input_fn` passed to `train`, `evaluate`,
                and `predict`. This should be a single `tf.Tensor` or `dict` of same.

            labels:
                This is the second item returned from the `input_fn` passed to `train`, `evaluate`,
                and `predict`. This should be a single `tf.Tensor` or `dict` of same
                (for multi-head models). If mode is `tf.estimator.ModeKeys.PREDICT`, `labels=None`
                will be passed. If the `model_fn`'s signature does not accept `mode`, the `model_fn`
                must still be able to handle `labels=None`.

            mode (tf.estimator.ModeKeys): Optional.
                Specifies if this training, evaluation or prediction. See `tf.estimator.ModeKeys`.

        Returns:
            `tf.estimator.EstimatorSpec`
        """
        spectrogram_length = features['spectrogram_length']
        spectrogram = features['spectrogram']

        # Create the inference graph.
        logits, seq_length = self.inference_fn(
            spectrogram, spectrogram_length, training=(mode == tf.estimator.ModeKeys.TRAIN))

        if mode == tf.estimator.ModeKeys.PREDICT:
            # CTC decode.
            decoded, plaintext, _ = self.decode_fn(logits, seq_length, None)

            prediction = {
                'decoded': tf.sparse.to_dense(decoded),
                'plaintext': plaintext
            }

            return tf.estimator.EstimatorSpec(mode=mode, predictions=prediction)

        tf.summary.scalar('spectrogram_length', spectrogram_length[0])
        tf.summary.image('spectrogram', tf.expand_dims(spectrogram, 3))

        # Convert dense labels tensor into sparse tensor.
        labels = tfc.layers.dense_to_sparse(labels)

        # CTC loss operator.
        self.loss_op = self.loss_fn(logits, seq_length, labels)

        # During training.
        if mode == tf.estimator.ModeKeys.TRAIN:
            # Set up the optimizer for training.
            global_step = tf.train.get_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                               beta1=FLAGS.adam_beta1, beta2=FLAGS.adam_beta2,
                                               epsilon=FLAGS.adam_epsilon)
            self.train_op = optimizer.minimize(loss=self.loss_op, global_step=global_step)

            # Add various hooks.
            self.hooks_fn()

        # Code for training and evaluation.
        label_plaintext = features['label_plaintext']

        # CTC decode.
        decoded, plaintext, plaintext_summary = self.decode_fn(logits,
                                                               seq_length,
                                                               label_plaintext)

        tf.summary.text('decoded_text', plaintext_summary[:, : FLAGS.num_samples_to_report])

        # Error metrics for decoded text.
        _, mean_ed, _, wer = self.error_rates_fn(labels, label_plaintext, decoded, plaintext)

        tf.summary.scalar('mean_edit_distance', mean_ed, family='Metrics')
        tf.summary.scalar('word_error_rate', wer, family='Metrics')

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=self.loss_op,
                                              train_op=self.train_op,
                                              training_hooks=self.hooks)

        # During evaluation.
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics_ops = {
                'mean_edit_distance': tf.metrics.mean(mean_ed, name='mean_edit_distance'),
                'word_error_rate': tf.metrics.mean(wer, name='word_error_rate')
            }

            return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss_op,
                                              eval_metric_ops=eval_metrics_ops)

        # This should never be reached.
        raise RuntimeError('Invalid mode.')

    @staticmethod
    def inference_fn(sequences, seq_length, training=True):
        """Build a TensorFlow inference graph according to the selected model in `FLAGS.used_model`.

        Supports the default [Deep Speech 1] model ('ds1') and an [Deep Speech 2] inspired
        implementation ('ds2').

        Args:
            sequences (tf.Tensor):
                3D float Tensor with input sequences. [batch_size, time, NUM_INPUTS]

            seq_length (tf.Tensor):
                1D int Tensor with sequence length. [batch_size]

            training (bool):
                If `True` apply dropout else if `False` the data is passed through unaltered.

        Returns:
            tf.Tensor: `logits`
                Softmax layer (logits) pre activation function, i.e. layer(X*W + b)
            tf.Tensor: `seq_length`
                1D Tensor containing approximated sequence lengths.
        """
        initializer = tf.truncated_normal_initializer(stddev=0.046875, dtype=TF_FLOAT)
        regularizer = tfc.layers.l2_regularizer(0.0046875)

        if FLAGS.used_model == 'ds1':
            # Dense input layers.
            output3 = tf_contrib.dense_layers(sequences, training, regularizer, initializer)
            # output3 = [batch_size, time, num_units_dense]

        elif FLAGS.used_model == 'ds2':
            # 2D convolutional input layers.
            with tf.variable_scope('conv'):
                # sequences = [batch_size, time, NUM_INPUTS] => [batch_size, time, NUM_INPUTS, 1]
                sequences = tf.expand_dims(sequences, 3)

                # Apply convolutions.
                output3, seq_length = tf_contrib.conv_layers(sequences)
        else:
            raise ValueError('Unsupported model "{}" in flags.'.format(FLAGS.used_model))

        # RNN layers.
        with tf.variable_scope('rnn'):
            rnn_dropout_rate = FLAGS.rnn_dropout_rate if training else 0.0

            if not FLAGS.cudnn:  # Use TensorFlow RNNs.
                # Create a stack of RNN cells.
                fw_cells, bw_cells = tf_contrib.bidirectional_cells(FLAGS.num_units_rnn,
                                                                    FLAGS.num_layers_rnn,
                                                                    dropout=rnn_dropout_rate)

                # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/stack_bidirectional_dynamic_rnn
                output_rnn, _, _ = tfc.rnn.stack_bidirectional_dynamic_rnn(
                    fw_cells, bw_cells,
                    inputs=output3,
                    dtype=TF_FLOAT,
                    sequence_length=seq_length,
                    parallel_iterations=64,
                    time_major=False
                )
                # output_rnn = [batch_size, time, num_units_rnn * 2]

            else:  # FLAGS.cudnn Use cuDNN RNNs.
                # cuDNN RNNs only support time major inputs.
                output3 = tfc.rnn.transpose_batch_time(output3)

                # https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnRNNRelu
                # https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnRNNTanh
                # https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnLSTM
                # https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnGRU
                supported_rnns = {
                    'rnn_relu': tfc.cudnn_rnn.CudnnRNNRelu,
                    'rnn_tanh': tfc.cudnn_rnn.CudnnRNNTanh,
                    'gru': tfc.cudnn_rnn.CudnnGRU,
                    'lstm': tfc.cudnn_rnn.CudnnLSTM
                }
                assert FLAGS.rnn_cell in supported_rnns

                rnn = supported_rnns[FLAGS.rnn_cell](num_layers=FLAGS.num_layers_rnn,
                                                     num_units=FLAGS.num_units_rnn,
                                                     input_mode='linear_input',
                                                     direction='bidirectional',
                                                     dropout=rnn_dropout_rate,
                                                     seed=FLAGS.random_seed,
                                                     dtype=TF_FLOAT,
                                                     # Glorot Uniform Initializer
                                                     kernel_initializer=None,
                                                     # Constant 0.0 Initializer
                                                     bias_initializer=None)

                output_rnn, _ = rnn(output3)
                output_rnn = tfc.rnn.transpose_batch_time(output_rnn)
                # output_rnn = [batch_size, time, num_units_rnn * 2]

        # Dense4
        with tf.variable_scope('dense4'):
            dense4 = tf.layers.dense(output_rnn, FLAGS.num_units_dense,
                                     activation=tf.nn.relu,
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer)
            dense4 = tf.minimum(dense4, FLAGS.relu_cutoff)
            dense4 = tf.layers.dropout(dense4, rate=FLAGS.dense_dropout_rate, training=training)
            # dense4 = [batch_size, conv_time, num_units_dense]

        # Logits: layer(XW + b),
        # We don't apply softmax here because most TensorFlow loss functions perform
        # a softmax activation as needed, and therefore don't expect activated logits.
        with tf.variable_scope('logits'):
            logits = tf.layers.dense(dense4, FLAGS.num_classes, kernel_initializer=initializer)
            logits = tfc.rnn.transpose_batch_time(logits)

        # logits = [time, batch_size, NUM_CLASSES]
        return logits, seq_length

    @staticmethod
    def loss_fn(logits, seq_length, labels):
        """Calculate the networks CTC loss.

        Args:
            logits (tf.Tensor):
                3D float Tensor. If time_major == False, this will be a Tensor shaped:
                [batch_size, max_time, num_classes]. If time_major == True (default), this will be a
                Tensor shaped: [max_time, batch_size, num_classes]. The logits.

            labels (tf.SparseTensor):
                An int32 SparseTensor. labels.indices[i, :] == [b, t] means labels.values[i] stores
                the id for (batch b, time t).

            seq_length (tf.Tensor):
                1D int32 vector, size [batch_size]. The sequence lengths.

        Returns:
            tf.Tensor: 1D float Tensor with size [1], containing the mean loss.
        """
        # https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss
        total_loss = tf.nn.ctc_loss(labels=labels,
                                    inputs=logits,
                                    sequence_length=seq_length,
                                    preprocess_collapse_repeated=False,
                                    ctc_merge_repeated=True,
                                    time_major=True)

        # Average CTC loss.
        mean_loss = tf.reduce_mean(total_loss)
        tf.summary.scalar('loss', mean_loss, family='Metrics')
        return mean_loss

    @staticmethod
    def decode_fn(logits, seq_len, originals=None):
        """Decode a given inference (`logits`) and convert it to plaintext.

        Args:
            logits (tf.Tensor):
                Logits Tensor of shape [time (input), batch_size, num_classes].

            seq_len (tf.Tensor):
                Tensor containing the batches sequence lengths of shape [batch_size].

            originals (tf.Tensor or None): Optional, default `None`.
                String Tensor of shape [batch_size] with the original plaintext.

        Returns:
            tf.Tensor: Decoded integer labels.
            tf.Tensor: Decoded plaintext's.
            tf.Tensor: Decoded plaintext's and original texts for comparision in `tf.summary.text`.
        """
        # tf.nn.ctc_beam_search_decoder provides more accurate results, but is slower.
        # https://www.tensorflow.org/api_docs/python/tf/nn/ctc_beam_search_decoder
        decoded, _ = tf.nn.ctc_beam_search_decoder(inputs=logits,
                                                   sequence_length=seq_len,
                                                   beam_width=FLAGS.beam_width,
                                                   top_paths=1,
                                                   merge_repeated=False)

        # ctc_greedy_decoder returns a list with one SparseTensor as only element, if `top_paths=1`.
        decoded = tf.cast(decoded[0], tf.int32)

        dense = tf.sparse.to_dense(decoded)

        originals = originals if originals is not None else np.array([], dtype=np.int32)

        # Translate decoded integer data back to character strings.
        plaintext, plaintext_summary = tf.py_func(metrics.dense_to_text, [dense, originals],
                                                  [tf.string, tf.string], name='py_dense_to_text')

        return decoded, plaintext, plaintext_summary

    @staticmethod
    def error_rates_fn(labels, originals, decoded, decoded_texts):
        """Calculate edit distance and word error rate.

        Args:
            labels (tf.SparseTensor or tf.Tensor):
                Integer SparseTensor containing the target.
                With dense shape [batch_size, time (target)].
                Dense Tensors are converted into SparseTensors if `FLAGS.use_warp_ctc == True`.

            originals (tf.Tensor):
                String Tensor of shape [batch_size] with the original plaintext.

            decoded (tf.Tensor):
                Integer tensor of the decoded output labels.

            decoded_texts (tf.Tensor)
                String tensor with the decoded output labels converted to normal text.

        Returns:
            tf.Tensor: Edit distances for the batch.
            tf.Tensor: Mean edit distance.
            tf.Tensor: Word error rates for the batch.
            tf.Tensor: Word error rate.
        """

        # Edit distances and average edit distance.
        edit_distances = tf.edit_distance(decoded, labels)
        mean_edit_distance = tf.reduce_mean(edit_distances)

        # Word error rates for the batch and average word error rate (WER).
        wers, wer = tf.py_func(metrics.wer_batch, [originals, decoded_texts],
                               [TF_FLOAT, TF_FLOAT], name='py_wer_batch')

        return edit_distances, mean_edit_distance, wers, wer

    def hooks_fn(self):
        """Produce and configure session hooks.

        Returns:
            List[tf.Tensor]: List containing TensorFlow hooks.
        """

        # GPU statistics hook.
        gpu_stats_hook = GPUStatisticsHook(
            log_every_n_steps=FLAGS.log_frequency,
            query_every_n_steps=FLAGS.gpu_hook_query_frequency,
            average_n=FLAGS.gpu_hook_average_queries,
            stats=['mem_util', 'gpu_util'],
            output_dir=FLAGS.train_dir,
            suppress_stdout=False,
            group_tag='gpu'
        )

        # Session hooks.
        self.hooks = [
            # Monitors the loss tensor and stops training if loss is NaN.
            tf.train.NanTensorHook(self.loss_op),
            gpu_stats_hook
        ]

        return self.hooks
