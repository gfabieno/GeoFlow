#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class trains the neural network
"""

import time
import tensorflow as tf
from vrmslearn.Inputqueue import BatchManager
from vrmslearn.RCNN2D import RCNN2D
from vrmslearn.Case import Case
from vrmslearn.DatasetGenerator import aggregate


class Trainer:
    """
    This class takes a NN model defined in tensorflow and performs the training
    """

    def __init__(self,
                 nn: RCNN2D,
                 case: Case,
                 checkpoint_dir: str = "./logs",
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 var_to_minimize: list = None):
        """
        Initialize the tester

        @params:
        nn (RCNN) : A tensforlow neural net
        case (Case) : A Case object on which to apply the NN
        checkpoint_dir (str): The path in which to save checkpoints
        learning_rate (float): The learning rate.
        beta1 (float): beta1 of the Adam optimizer
        beta2 (float): beta2 of the Adam optimizer
        epsilon (float): epsilon of the Adam optimizer
        var_to_minimize (list): A list of tf.Variables to optimize when training

        @returns:
        """
        self.nn = nn
        self.case = case
        self.checkpoint_dir = checkpoint_dir
        with self.nn.graph.as_default():
            self.global_step = tf.train.get_or_create_global_step()

            # Output the graph for Tensorboard
            writer = tf.summary.FileWriter(self.checkpoint_dir,
                                           graph=tf.get_default_graph())
            writer.close()
            self.tomin = self.define_optimizer(learning_rate,
                                               beta1,
                                               beta2,
                                               epsilon,
                                               var_to_minimize)

    def define_optimizer(self,
                         learning_rate: float = 0.001,
                         beta1: float = 0.9,
                         beta2: float = 0.999,
                         epsilon: float = 1e-8,
                         var_to_minimize: list = None):
        """
        This method creates an optimization node using Adam optimizer.

        @params:
        learning_rate (float): The learning rate.
        beta1 (float): beta1 of the Adam optimizer
        beta2 (float): beta2 of the Adam optimizer
        epsilon (float): epsilon of the Adam optimizer

        @returns:
        tomin (tf.tensor) : Output of the optimizer node.
        """
        with tf.name_scope("Optimizer"):
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                         beta1=beta1,
                                         beta2=beta2,
                                         epsilon=epsilon,
                                         name="Adam")

            # Add node to minimize loss
            if var_to_minimize:
                tomin = opt.minimize(self.nn.loss,
                                     global_step=self.global_step,
                                     var_list=var_to_minimize)
            else:
                tomin = opt.minimize(self.nn.loss, global_step=self.global_step)

        return tomin

    def train_model(self,
                    niter: int = 10,
                    restore_from: str = None,
                    thread_read: int = 1):
        """
        This method trains the model. The training is restarted automatically
        if any checkpoints are found in self.checkpoint_dir.

        @params:
        niter (int) : Number of total training iterations to run.
        restore_from (str): Checkpoint file from which to initialize parameters
        thread_read (int): Number of threads to create example by InputQueue

        @returns:
        """

        # Print optimizer settings being used, batch size, niter
        print("number of iterations (niter) = " + str(niter))

        # Do the learning
        generator_fun = [self.case.get_example] * thread_read
        with BatchManager(batch_size=self.nn.batch_size,
                          generator_fun=generator_fun,
                          postprocess_fun=self.case.ex2batch) as batch_queue:

            with self.nn.graph.as_default():
                summary_op = tf.summary.merge_all()

                # The StopAtStepHook handles stopping after running given steps.
                hooks = [tf.train.StopAtStepHook(last_step=niter),
                         tf.train.SummarySaverHook(save_steps=10,
                                                   summary_op=summary_op),
                         tf.train.CheckpointSaverHook(
                             checkpoint_dir=self.checkpoint_dir,
                             save_steps=100,
                             saver=tf.train.Saver(max_to_keep=None))]

                if restore_from is not None:
                    saver = tf.train.Saver(tf.trainable_variables())
                    with tf.Session() as sess:
                        saver.restore(sess, restore_from)
                        vars = tf.trainable_variables()
                        trained_variables = sess.run(vars)

                    assigns = [tf.assign(v, trained_variables[ii])
                               for ii, v in enumerate(vars)]

                # Run the training iterations
                with tf.train.MonitoredTrainingSession(checkpoint_dir=self.checkpoint_dir,
                                                       save_checkpoint_secs=None,
                                                       save_summaries_steps=1,
                                                       hooks=hooks) as sess:

                    if restore_from is not None:
                        batch = batch_queue.next_batch()
                        feed_dict = {self.nn.feed_dict[lbl]: batch[lbl]
                                     for lbl in self.nn.feed_dict}
                        step = sess.run(self.global_step, feed_dict=feed_dict)
                        if step == 0:
                            sess.run(assigns, feed_dict=feed_dict)

                    while not sess.should_stop():
                        t0 = time.time()
                        batch = batch_queue.next_batch()
                        t1 = time.time()

                        feed_dict = {self.nn.feed_dict[lbl]: batch[lbl]
                                     for lbl in self.nn.feed_dict}
                        step, loss, _ = sess.run([self.global_step,
                                                  self.nn.loss,
                                                  self.tomin],
                                                  feed_dict=feed_dict)
                        t2 = time.time()
                        print("Iteration %d, loss: %f, t_batch: %f, t_graph: %f , nqueue: %d" % (step, loss, t1 - t0, t2 - t1,
                                                  batch_queue.n_in_queue.value))

    def evaluate(self, toeval, niter, checkpoint_dir=None, batch=None):
        """
        This method compute outputs contained in toeval of a NN.

        @params:
        niter (int) : Training iterations of the checkpoint
        checkpoint_dir (str): Directory of the checkpoint
        batch (tuple): A batch as created by batch_generator, to predict from

        @returns:
        evaluated (list): A list of np.array containing evalutead tensors
        """

        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir

        feed_dict = dict(zip(self.nn.feed_dict, batch))

        with self.nn.graph.as_default():
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess,
                              checkpoint_dir + '/model.ckpt-' + str(niter))
                evaluated = sess.run(toeval, feed_dict=feed_dict)

        return evaluated
