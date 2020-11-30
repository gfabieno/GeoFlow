# -*- coding: utf-8 -*-
"""
Define custom losses.
"""

import tensorflow as tf


def ref_loss(use_weights=True):
    """
    Get the loss function for the reflection prediction.
    """
    def loss(label, output):
        label, weights = label[:, 0], label[:, 1]
        #  Logistic regression of zero offset time of reflections
        weights = tf.expand_dims(weights, -1)
        if not use_weights:
            weights = tf.ones_like(weights)
        output = output * weights
        temp_lbl = tf.cast(label, tf.int32)
        label = tf.one_hot(temp_lbl, 2) * weights

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=output,
            labels=label))
        return loss

    return loss


def v_compound_loss(alpha=0.2, beta=0.1, use_weights=True):
    """
    Get the three-part loss function for velocity.

    :param alpha: Proportion of loss associated with minimizing the error
                  between derivatives.
    :param beta: Proportion of loss associated with minimizing the gradient
                 (blocky inversion).

    :return: Output of node calculating loss.
    """
    fact1 = 1 - alpha - beta

    def loss(label, output):
        label, weight = label[:, 0], label[:, 1]
        if not use_weights:
            weight = tf.ones_like(weight)
        output = output[:, :, :, 0]
        losses = []

        # Calculate mean squared error
        if fact1 > 0:
            num = tf.reduce_sum(weight * (label-output)**2)
            den = tf.reduce_sum(weight * label**2)
            losses.append(fact1 * num / den)

        #  Calculate mean squared error of the z derivative
        if alpha > 0:
            dlabel = label[:, 1:, :] - label[:, :-1, :]
            dout = output[:, 1:, :] - output[:, :-1, :]
            num = tf.reduce_sum(weight[:, :-1, :] * (dlabel-dout)**2)
            den = tf.reduce_sum(weight[:, :-1, :] * dlabel**2) + 1E-6
            losses.append(alpha * num / den)

        # Minimize gradient (blocky inversion)
        if beta > 0:
            num = (
                tf.norm(output[:, 1:, :] - output[:, :-1, :], ord=1)
                + tf.norm(output[:, :, 1:] - output[:, :, :-1], ord=1)
            )
            den = tf.norm(output, ord=1) / 0.02
            losses.append(beta * num / den)

        return tf.reduce_sum(losses)

    return loss
