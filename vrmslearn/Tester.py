#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class tests a NN on a dataset.
"""
import fnmatch
import os

import h5py as h5
import tensorflow as tf

from vrmslearn.RCNN import RCNN
from vrmslearn.Case import Case
from vrmslearn.DatasetGenerator import aggregate

class Tester(object):
    """
    This class tests a NN on a dataset.
    """

    def __init__(self,
                 NN: RCNN,
                 case: Case):
        """
        Initialize the tester

        @params:
        NN (RCNN) : A tensforlow neural net
        data_generator (SeismicGenerator): A data generator object

        @returns:
        """
        self.NN = NN
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
        filename (str): The structure of the examples' filenames
        toeval (list): List of tensors to predict
        restore_from (str): File containing the trained weights

        @returns:
        """

        predictions = fnmatch.filter(os.listdir(savepath), filename)
        predictions = [os.path.join(savepath, p) for p in predictions]
        with self.NN.graph.as_default():
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

                    if len(batch) == self.NN.batch_size:
                        batch = aggregate(batch)
                        feed_dict = dict(zip(self.NN.feed_dict, batch))
                        evaluated = sess.run(toeval, feed_dict=feed_dict)
                        for jj, bexample in enumerate(bexamples):
                            savefile = h5.File(bexample, "w")
                            for kk, el in enumerate(toeval_names):
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
        maskname(str) : name of the valid predictions mask
        savepath (str) : The path in which the test examples are found
        filename (str): The structure of the examples' filenames
        tester_n (int): Label for the model to use for prediction

        @returns:
        labels (list): List containing all labels
        preds (list):  List containing all predictions
        """
        
        predictions = fnmatch.filter(os.listdir(savepath), filename)
        predictions = [os.path.join(savepath, p) for p in predictions]
        labels = []
        preds = []
        lfiles = []
        pfiles = []
        for ii, example in enumerate(self.case.files["test"]):
            example_pred = os.path.basename(example)
            if example_pred in predictions:
                pfiles.append(example_pred)
                pfile = h5.File(pfiles[-1], "r")
                preds.append(pfile[predname][:])
                pfile.close()
                lfiles.append(example)
                lfile = h5.File(lfiles[-1], "r")
                labels.append(lfile[labelname][:])
                lfile.close()

        return labels, preds, lfiles, pfiles
