"""Utility and helper methods for TensorFlow speech learning."""

import tensorflow as tf


class AdamOptimizerLogger(tf.train.AdamOptimizer):
    # TODO: Document
    def _apply_dense(self, grad, var):
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta1_power, beta2_power = self._get_beta_accumulators()

        m_hat = m / (1. - beta1_power)
        v_hat = v / (1. - beta2_power)

        step = m_hat / (v_hat ** 0.5 + self._epsilon_t)

        # Use a histogram summary to monitor it during training.
        tf.summary.histogram('hist', step)

        current_lr = self._lr_t * tf.sqrt(1. - beta2_power) / (1. - beta1_power)
        tf.summary.scalar('estimated_lr', current_lr)

        return super(AdamOptimizerLogger, self)._apply_dense(grad, var)
