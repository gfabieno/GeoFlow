#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Keras data input sequence."""

from tensorflow.keras.utils import Sequence
import numpy as np

PHASE_DICT = {
    True: "train",
    False: "test",
}

OUTS = ('ref', 'vrms', 'vint', 'vdepth')


class Sequence(Sequence):
    def __init__(
                self,
                is_training,
                case,
                batch_size,
                input_size,
                depth_size,
                out_names,
            ):
        self.is_training = is_training
        self.phase = PHASE_DICT[self.is_training]
        self.case = case

        self.batch_size = batch_size
        self.input_size = input_size
        self.depth_size = depth_size

        for lbl in out_names:
            if lbl not in OUTS:
                raise ValueError(f"`out_names` should be from {OUTS}")
        self.out_names = [name for name in OUTS if name in out_names]

        gen = self.case.sample_generator.model_gen
        self.data_names = ["input", *gen.label_names, *gen.weight_names]

        if not self.is_training:
            self.reset_test_generator()

    def __len__(self):
        return len(self.case.files[self.phase])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, _):
        inputs = np.empty([self.batch_size, *self.input_size])
        labels = []

        n_t = self.input_size[0]
        n_cmp = self.input_size[2]
        LABEL_SHAPE = {
            'ref': [self.batch_size, 2, n_t, n_cmp],
            'vrms': [self.batch_size, 2, n_t, n_cmp],
            'vint': [self.batch_size, 2, n_t, n_cmp],
            'vdepth': [self.batch_size, 2, n_t, n_cmp],
        }
        for lbl in self.out_names:
            labels.append(np.empty(LABEL_SHAPE[lbl]))

        WEIGHT_MAPPING = {
            'ref': 'tweight',
            'vrms': 'tweight',
            'vint': 'tweight',
            'vdepth': 'dweight',
        }
        if not self.is_training:
            filenames = []
        for i in range(self.batch_size):
            if self.is_training:
                data = self.case.get_example(phase=self.phase)
            else:
                try:
                    filename = next(self.test_filenames_generator)
                    data = self.case.get_example(
                        filename=filename,
                        phase=self.phase,
                    )
                    filenames.append(filename)
                except StopIteration:
                    break

            inputs[i] = data[0]
            for j, lbl in enumerate(self.out_names):
                label_idx = self.data_names.index(lbl)
                weight_idx = self.data_names.index(WEIGHT_MAPPING[lbl])
                labels[j][i] = [data[label_idx][:n_t], data[weight_idx][:n_t]]

        if self.is_training:
            return inputs, labels
        else:
            return inputs, filenames

    def reset_test_generator(self):
        self.test_filenames_generator = (
            f for f in self.case.files[self.phase]
        )  # This is a generator. Extract the next value with `next`.
