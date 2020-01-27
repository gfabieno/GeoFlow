#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class tests a NN on a dataset.
"""
import fnmatch
import os

import h5py as h5
import tensorflow as tf

from vrmslearn.RCNN2D import RCNN2D
from vrmslearn.Case import Case
from vrmslearn.DatasetGenerator import aggregate

class Tester(object):
    """
    This class tests a NN on a dataset.
    """

    def __init__(self,
                 nn: RCNN2D,
                 case: Case):
        """
        Initialize the tester

        @params:
        nn (RCNN) : A tensforlow neural net
        data_generator (SeismicGenerator): A data generator object

        @returns:
        """
        self.nn = nn
        self.case = case

    def test_dataset(self,
                     savepath: str,
                     toeval: list,
                     toeval_names: list,
                     filename: str = 'example_*',
                     restore_from: str = None):
        """
        This method evaluate predictions on all examples contained in savepath,
        and save the predictions in hdf5 files.

        @params:
        savepath (str) : The path in which the test examples are found
        toeval (list): List of tensors to predict
        toeval_names (list): List of strings with the name of tensors to predict
        filename (str): The structure of the examples' filenames
        restore_from (str): File containing the trained weights

        @returns:
        """

        predictions = fnmatch.filter(os.listdir(savepath), filename)
        predictions = [os.path.join(savepath, p) for p in predictions]
        with self.nn.graph.as_default():
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, restore_from)
                batch = []
                bexamples = []
                for ii, example in enumerate(self.case.files["test"]):
                    predname = os.path.basename(example)
                    if predname not in predictions:
                        bexamples.append(example)
                        batch.append(self.case.get_example(filename=example))

                    if len(batch) == self.nn.batch_size:
                        batch = aggregate(batch)
                        feed_dict = dict(zip(self.nn.feed_dict, batch))
                        evaluated = sess.run(toeval, feed_dict=feed_dict)
                        for jj, bexample in enumerate(bexamples):
                            savefile = h5.File(bexample, "r+")
                            for kk, el in enumerate(toeval_names):
                                if el in savefile.keys():
                                    del savefile[el]
                                savefile[el] = evaluated[kk][jj, :]
                            savefile.close()
                        batch = []
                        bexamples = []

    def get_preds(self,
                  labelname: str,
                  predname: str,
                  savepath: str,
                  filename: str = 'example_*'):
        """
        This method returns the labels and the predictions for an output.

        @params:
        labelname (str) : Name of the labels in the example file
        predname (str) : Name of the predictions in the example file
        savepath (str) : The path in which the test examples are found
        filename (str): The structure of the examples' filenames

        @returns:
        labels (list): List containing all labels
        preds (list):  List containing all predictions
        """
        labels = []
        preds = []
        for ii, example in enumerate(self.case.files["test"]):
            file = h5.File(example, "r")
            if predname in file.keys():
                preds.append(file[predname][:])
                labels.append(file[labelname][:])
            file.close()

        return labels, preds

