#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Keras data input sequence."""

from tensorflow.keras.utils import Sequence
from vrmslearn.Dataset import Dataset
from typing import List

OUTS = ('ref', 'vrms', 'vint', 'vdepth')


class Sequence(Sequence):
    def __init__(self,
                 is_training: bool,
                 dataset: Dataset,
                 batch_size: int,
                 tooutputs: List[str],
                 toinput: str):
        """
        Create a tf.keras.Sequence from a Dataset object

        :param is_training: If true, use thes training set, else uses the test
        :param dataset:        A Dataset object for generating examples
        :param batch_size:  The batch size
        :param tooutputs:   The list of the name of the desired outputs
        :param toinput:     The name of the input
        """

        self.is_training = is_training
        if is_training:
            self.phase = "train"
        else:
            self.phase = "test"
        self.dataset = dataset
        self.batch_size = batch_size
        self.toinput = toinput
        self.tooutputs = tooutputs
        self.dataset._getfilelist(phase=self.phase)
        self.len = int(len(self.dataset.files[self.phase])/self.batch_size)

    def __len__(self):
        return self.len

    # TODO is that necessary to redefine this ?
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, _):
        inputs, labels, files = self.dataset.get_batch(self.batch_size,
                                                    phase=self.phase,
                                                    toinputs=self.toinput,
                                                    tooutputs=self.tooutputs)
        # TODO Is a different signature necessary here ?
        if self.is_training:
            return inputs, labels
        else:
            return inputs, files
