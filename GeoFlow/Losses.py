# -*- coding: utf-8 -*-
"""
Define custom losses.
"""

import tensorflow as tf


def ref_loss():
    """
    Get the loss function for the reflection prediction.
    """
    def loss(label, output):
        label, weight = label[:, 0], label[:, 1]
        # Logistic regression of zero offset time of reflections.
        weight = tf.expand_dims(weight, -1)
        output = output * weight
        temp_lbl = tf.cast(label, tf.int32)
        label = tf.one_hot(temp_lbl, 2) * weight

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=output,
                                                       labels=label)
        loss = tf.reduce_mean(loss, axis=[1, 2])
        return loss

    return loss


def v_compound_loss(alpha=0.2, beta=0.1):
    """
    Get the three-part loss function for velocity.

    :param alpha: Proportion of loss associated with minimizing the error
                  between derivatives.
    :param beta: Proportion of loss associated with minimizing the gradient
                 (blocky inversion).

    :return: Output of node calculating loss.
    """
    assert alpha >= 0 and alpha <= 1
    assert beta >= 0 and beta <= 1
    assert beta + alpha >= 0 and beta + alpha <= 1
    fact1 = 1 - alpha - beta

    def loss(label, output):
        label, weight = label[:, 0], label[:, 1]
        output = output[..., 0]
        losses = []

        # Compute mean squared error.
        if fact1 > 0:
            num = tf.reduce_sum(weight * (label-output)**2, axis=[1, 2])
            den = tf.reduce_sum(weight * label**2, axis=[1, 2])
            losses.append(fact1 * num / den)

        # Compute mean squared error of the vertical derivative.
        if alpha > 0:
            dlabel = label[:, 1:, :] - label[:, :-1, :]
            dout = output[:, 1:, :] - output[:, :-1, :]
            num = tf.reduce_sum(weight[:, :-1, :] * (dlabel-dout)**2,
                                axis=[1, 2])
            den = tf.reduce_sum(weight[:, :-1, :] * dlabel**2,
                                axis=[1, 2]) + 1E-6
            losses.append(alpha * num / den)

        # Minimize gradient (blocky inversion).
        if beta > 0:
            num = tf_norm(output[:, 1:, :] - output[:, :-1, :],
                          axis=[1, 2])
            if output.get_shape()[-1] != 1:
                num += tf_norm(output[:, :, 1:] - output[:, :, :-1],
                               axis=[1, 2])
            den = tf.norm(output, ord=1, axis=[1, 2]) / .02
            losses.append(beta * num / den)

        return tf.reduce_sum(losses, axis=0)

    return loss


def tf_norm(input, axis=None):
    return tf.reduce_sum(tf.abs(input), axis=axis)
