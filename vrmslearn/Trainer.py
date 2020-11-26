#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class trains the neural network
"""

from os.path import join, split

import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

from vrmslearn.RCNN2D import RCNN2D
from vrmslearn.Sequence import Sequence, OUTS

WEIGHTS_NAME = "{epoch:04d}.ckpt"


class Trainer:
    """
    This class takes a NN model defined in tensorflow and performs the training
    """

    def __init__(self,
                 nn: RCNN2D,
                 sequence: Sequence,
                 checkpoint_dir: str = "./logs",
                 learning_rate: float = 0.001,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-8,
                 loss_scales: dict = {'ref': 1.},
                 use_weights: bool = True):
        """
        Initialize the tester

        @params:
        nn (RCNN) : A tensorflow neural net
        sequence (Sequence) : A Sequence object providing data
        checkpoint_dir (str): The path in which to save checkpoints
        learning_rate (float): The learning rate.
        beta1 (float): beta1 of the Adam optimizer
        beta2 (float): beta2 of the Adam optimizer
        epsilon (float): epsilon of the Adam optimizer
        loss_scales (dict): losses associated with each label
        use_weights (bool): whether to use weights or not in losses.
        """
        self.nn = nn
        self.sequence = sequence
        self.checkpoint_dir = checkpoint_dir

        for lbl in loss_scales.keys():
            if lbl not in OUTS:
                raise ValueError(f"`loss_scales` keys should be from {OUTS}")
        self.loss_scales = loss_scales
        self.out_names = [out for out in OUTS if out in loss_scales.keys()]

        losses, losses_weights = [], []
        for lbl in self.out_names:
            if lbl == 'ref':
                losses.append(ref_loss(use_weights=use_weights))
            else:
                loss = v_compound_loss(use_weights=use_weights)
                losses.append(loss)
            losses_weights.append(self.loss_scales[lbl])

        self.nn.compile(
            optimizer=optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
                name="Adam",
            ),
            loss=losses,
            loss_weights=losses_weights,
        )

    def train_model(self,
                    epochs: int = 5,
                    steps_per_epoch: int = 100,
                    restore_from: str = None):
        """
        This method trains the model. The training is restarted automatically
        if any checkpoints are found in self.checkpoint_dir.

        @params:
        epochs (int): quantity of epochs, of `steps_per_epoch` iterations
        steps_per_epoch (int): quantity of iterations per epoch
        restore_from (str): Checkpoint file from which to initialize parameters
        """
        if restore_from is not None:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                self.nn.load_weights(restore_from)
            filename = split(restore_from)[-1]
            initial_epoch = int(filename[:4])
            epochs += initial_epoch
        else:
            initial_epoch = 0

        tensorboard = callbacks.TensorBoard(log_dir=self.checkpoint_dir)
        checkpoints = callbacks.ModelCheckpoint(join(self.checkpoint_dir,
                                                     WEIGHTS_NAME),
                                                save_weights_only=True,
                                                save_freq='epoch')
        self.nn.fit(self.sequence,
                    epochs=epochs,
                    callbacks=[tensorboard, checkpoints],
                    initial_epoch=initial_epoch,
                    steps_per_epoch=steps_per_epoch,
                    max_queue_size=10,
                    use_multiprocessing=False)


def ref_loss(use_weights=True):
    """Get the loss function for the reflection prediction."""
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
    """Get the three-part loss function for velocity.

    @params:
        alpha (scalar) : proportion of loss associated with minimizing the
                         error between derivatives
        beta (scalar) : proportion of loss associated with minimizing the
                         gradient (blocky inversion)

    @returns:
        loss (tf.tensor) : Output of node calculating loss.
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
