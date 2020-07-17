#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class trains the neural network
"""

from os.path import join

import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

from vrmslearn.RCNN2D import RCNN2D
from vrmslearn.Sequence import Sequence, OUTS


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
                 loss_scales: dict = {'ref': 1.}):
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

        @returns:
        """
        self.nn = nn
        self.sequence = sequence
        self.checkpoint_dir = checkpoint_dir

        self.loss_scales = loss_scales
        for lbl in loss_scales.keys():
            if lbl not in OUTS:
                raise ValueError(f"`loss_scales` keys should be from {OUTS}")
        losses, losses_weights = [], []
        for lbl in OUTS:
            if lbl in self.loss_scales.keys():
                if lbl == 'ref':
                    losses.append(ref_loss())
                else:
                    losses.append(v_compound_loss())
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
                    batch_size: int = 1,
                    epochs: int = 5,
                    steps_per_epoch: int = 100,
                    restore_from: str = None,
                    thread_read: int = 1):
        """
        This method trains the model. The training is restarted automatically
        if any checkpoints are found in self.checkpoint_dir.

        @params:
        batch_size (int): quantity of examples per batch
        epochs (int): quantity of epochs, of `steps_per_epoch` iterations
        steps_per_epoch (int): quantity of iterations per epoch
        restore_from (str): Checkpoint file from which to initialize parameters
        thread_read (int): Number of threads to create example by InputQueue
        """
        if restore_from is not None:
            self.nn.load_weights(
                join(self.checkpoint_dir, restore_from)
            )

        tensorboard = callbacks.TensorBoard(log_dir=self.checkpoint_dir)
        checkpoints = callbacks.ModelCheckpoint(
            self.checkpoint_dir,
            save_weights_only=True,
            save_freq='epoch',
        )
        self.nn.fit(
            self.sequence,
            epochs=epochs,
            callbacks=[tensorboard, checkpoints],
            # initial_epoch=0,
            steps_per_epoch=steps_per_epoch,
            max_queue_size=10,
            workers=10,
            use_multiprocessing=True,
        )


def ref_loss():
    """Get the loss function for the reflection prediction."""
    def loss(label, output):
        label, weights = label
        #  Logistic regression of zero offset time of reflections
        weights = tf.expand_dims(weights, -1)
        # if self.with_masking:
        #     weightsr = tf.expand_dims(self.weights['wtime'], -1)
        # else:
        #     weightsr = 1.0
        output = output * weights
        temp_lbl = tf.cast(label, tf.int32)
        label = tf.one_hot(temp_lbl, 2) * weights

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=output,
                labels=label,
            )
        )
        return loss

    return loss


def v_compound_loss(alpha=0.2, beta=0.1):
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
        label, weight = label
        losses = []

        # Calculate mean squared error
        if fact1 > 0:
            num = tf.reduce_sum(weight * (label-output)**2)
            den = tf.reduce_sum(weight * label**2)
            losses.append(fact1 * num / den)

        #  Calculate mean squared error of the z derivative
        if alpha > 0:
            dlabel = label[1:, :] - label[:-1, :]
            dout = output[1:, :] - output[:-1, :]
            num = tf.reduce_sum(weight[:-1, :] * (dlabel-dout)**2)
            den = tf.reduce_sum(weight[:-1, :] * dlabel**2 + 1E-6)
            losses.append(alpha * num / den)

        # Minimize gradient (blocky inversion)
        if beta > 0:
            num = (
                tf.norm(output[1:, :] - output[:-1, :], ord=1)
                + tf.norm(output[:, 1:] - output[:, :-1], ord=1)
            )
            den = tf.norm(output, ord=1) / 0.02
            losses.append(beta * num / den)

        return tf.reduce_sum(losses)

    return loss
