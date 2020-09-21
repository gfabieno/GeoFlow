#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class tests a NN on a dataset.
"""

import fnmatch
import os
from os.path import join, basename

import h5py as h5
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np

from vrmslearn.RCNN2D import RCNN2D
from vrmslearn.Case import Case
from vrmslearn.Sequence import Sequence


class Tester(object):
    """
    This class tests a NN on a dataset.
    """

    def __init__(self,
                 nn: RCNN2D,
                 sequence: Sequence,
                 case: Case):
        """
        Initialize the tester

        @params:
        nn (RCNN) : A tensforlow neural net
        sequence (Sequence) : A Sequence object providing data
        """
        self.nn = nn
        self.sequence = sequence
        self.case = case

        self.out_names = self.nn.out_names

    def test_dataset(self,
                     savepath: str,
                     restore_from: str = None):
        """
        This method evaluate predictions on all examples contained in savepath,
        and save the predictions in hdf5 files.

        @params:
        savepath (str) : The path in which the test examples are found
        toeval (dict): Dict of name: tensors to predict
        filename (str): The structure of the examples' filenames
        batch_size (int): quantity of examples per batch
        restore_from (str): File containing the trained weights

        @returns:
        """
        if restore_from is not None:
            self.nn.load_weights(restore_from)

        self.sequence.reset_test_generator()

        for data, filenames in self.sequence:
            evaluated = self.nn.predict(
                data,
                max_queue_size=10,
                use_multiprocessing=False,
            )
            is_batch_incomplete = len(data) != len(filenames)
            if is_batch_incomplete:
                for i in range(len(evaluated)):
                    evaluated[i] = evaluated[i][:len(filenames)]

            for i, (lbl, out) in enumerate(zip(self.out_names, evaluated)):
                if lbl != 'ref':
                    evaluated[i] = out[..., 0]

            for i, example in enumerate(filenames):
                example = join(savepath, basename(example))
                with h5.File(example, "w") as savefile:
                    for j, el in enumerate(self.out_names):
                        if el in savefile.keys():
                            del savefile[el]
                        savefile[el] = evaluated[j][i, :]

    def get_preds(self,
                  prednames: list,
                  savepath: str,
                  examples: list = None,
                  filename: str = 'example_*'):
        """
        This method returns the labels and predictions for labels in prednames.

        @params:
        prednames (list) : List of name of the predictions in the example file
        savepath (str) : The path in which the predictions are found
        examples (list):   List of name of example to get predictions.
                           If None, predictions for all examples in savepath
                           are returned
        filename (str): The structure of the examples' filenames

        @returns:
        labels (dict):  Dict containing the {predname: label}
        preds (dict):   Dict containing the {predname: prediction}
        """
        preds = {predname: [] for predname in prednames}
        labels = []
        if examples is None:
            examples = fnmatch.filter(os.listdir(savepath), filename)

        for ii, example in enumerate(examples):
            file = h5.File(os.path.join(savepath, example), "r")
            for predname in prednames:
                if predname in file.keys():
                    preds[predname].append(file[predname][:])
            file.close()
            labelfile = os.path.join(self.case.datatest, example)
            labels.append(self.case.get_example(labelfile))

        labels = self.case.ex2batch(labels)

        return labels, preds

    def plot_predictions(self,
                         labelnames: list,
                         savepath: str,
                         quantity: int = 1,
                         image=True):
        """
        This method plots the labels and the predictions for each test sample.

        @params:
        labelnames (list) : List of names of the labels in the example file
        savepath (str) : The path in which the test examples are found
        """

        examples = [os.path.basename(self.case.files["test"][ii])
                    for ii in range(quantity)]

        labels, preds = self.get_preds(labelnames, savepath, examples=examples)
        for ii in range(quantity):
            if image:
                fig, axes = plt.subplots(len(labelnames), 2)
            else:
                fig, axes = plt.subplots(1, len(labelnames), squeeze=False)

            labeld = {la: labels[la][ii] for la in labelnames}
            predd = {la: preds[la][ii] for la in labelnames}
            label, pred = self.case.label.postprocess(labeld, predd)
            for jj, labelname in enumerate(labelnames):
                if image:
                    vmin = np.min(label[labelname])
                    vmax = np.max(label[labelname])
                    axes[jj, 0].imshow(
                        label[labelname],
                        vmin=vmin,
                        vmax=vmax,
                        cmap='inferno',
                        aspect='auto',
                    )
                    axes[jj, 1].imshow(
                        pred[labelname],
                        vmin=vmin,
                        vmax=vmax,
                        cmap='inferno',
                        aspect='auto',
                    )

                else:
                    y = np.arange(label[labelname].shape[0])
                    axes[0, jj].plot(label[labelname][:, 0], y)
                    axes[0, jj].plot(pred[labelname][:, 0], y)
                    axes[0, jj].invert_yaxis()

            plt.show()

    def animated_predictions(self,
                             labelnames: list,
                             savepath: str,
                             quantity: int = None,
                             image: bool = True):
        """
        Makes an animation that shows iteratively the data, labels and
        predictions over the testing dataset.

        @params:
        labelnames (list) : List of names of the labels to predict
        savepath (str) : The path in which the test examples are found
        quantity (int): Number of examples to show. If None, show all examples
                        in the test set
        image (bool):   If True, labels and predictions are shown as images,
                        else plot 1D profiles.
        """
        if quantity is None:
            examples = [
                os.path.basename(f) for f in self.case.files["test"]
                if os.path.basename(f) in os.listdir(savepath)
            ]
        else:
            examples = [
                os.path.basename(self.case.files["test"][ii])
                for ii in range(quantity)
            ]

        labels, preds = self.get_preds(
            labelnames, savepath, examples=examples,
        )
        datas = labels['input']
        datas = [np.reshape(el, [el.shape[0], -1]) for el in datas]

        if image:
            fig, axs = plt.subplots(2, 1 + len(labelnames), squeeze=False)
        else:
            fig, axs = plt.subplots(1, 1 + len(labelnames), squeeze=False)

        clip = 0.01
        vmax = np.max(datas) * clip
        vmin = -vmax
        im1 = axs[0, 0].imshow(
            datas[0],
            animated=True,
            vmin=vmin,
            vmax=vmax,
            aspect='auto',
            cmap=plt.get_cmap('Greys'),
        )
        axs[0, 0].set_title('data')
        ims = [im1]

        labeld = {la: labels[la][0] for la in labelnames}
        predd = {la: preds[la][0] for la in labelnames}
        label, pred = self.case.label.postprocess(labeld, predd)

        for ii, labelname in enumerate(labelnames):
            if image:
                im1 = axs[0, 1 + ii].imshow(
                    pred[labelname],
                    vmin=0,
                    vmax=1,
                    animated=True,
                    cmap='inferno',
                    aspect='auto',
                )
                im2 = axs[1, 1 + ii].imshow(label[labelname], vmin=0, vmax=1,
                                            animated=True,
                                            cmap='inferno', aspect='auto')
                axs[0, 1 + ii].set_title(labelname)
                plt.colorbar(
                    im1,
                    ax=axs[0, 1 + ii],
                    orientation="horizontal",
                    pad=0.15,
                    fraction=0.1,
                )
                plt.colorbar(
                    im2,
                    ax=axs[1, 1 + ii],
                    orientation="horizontal",
                    pad=0.15,
                    fraction=0.1,
                )
                ims.append(im1)
                ims.append(im2)
            else:
                y = np.arange(pred[labelname].shape[0])

                im1, = axs[0, 1 + ii].plot(label[labelname][:, 0][:len(y)], y)
                im2, = axs[0, 1 + ii].plot(pred[labelname][:, 0][:len(y)], y)
                axs[0, 1 + ii].set_ylim(np.min(y), np.max(y))
                axs[0, 1 + ii].set_xlim(-0.1, 1.1)
                axs[0, 1 + ii].invert_yaxis()
                axs[0, 1 + ii].set_title(labelname)
                ims.append(im1)
                ims.append(im2)
        plt.tight_layout()

        def init():
            for ii, im in enumerate(ims):
                labeld = {la: labels[la][0] for la in labelnames}
                predd = {la: preds[la][0] for la in labelnames}
                label, pred = self.case.label.postprocess(labeld, predd)
                if ii == 0:
                    toplot = datas[0]
                    im.set_array(toplot)
                else:
                    if ii % 2 == 0:
                        toplot = pred[labelnames[(ii - 1) // 2]]
                    else:
                        toplot = label[labelnames[(ii - 1) // 2]]
                    if image:
                        im.set_array(toplot)
                    else:
                        y = np.arange(toplot.shape[0])
                        im.set_data(toplot, y)
            return ims

        def animate(t):
            labeld = {la: labels[la][t] for la in labelnames}
            predd = {la: preds[la][t] for la in labelnames}
            label, pred = self.case.label.postprocess(labeld, predd)

            for ii, im in enumerate(ims):
                if ii == 0:
                    toplot = datas[t]
                    im.set_array(toplot)
                else:
                    if ii % 2 == 0:
                        toplot = pred[labelnames[(ii - 1) // 2]]
                    else:
                        toplot = label[labelnames[(ii - 1) // 2]]
                    if image:
                        im.set_array(toplot)
                    else:
                        y = np.arange(toplot.shape[0])
                        im.set_data(toplot, y)
            return ims

        _ = animation.FuncAnimation(fig,
                                    animate,
                                    init_func=init,
                                    frames=len(datas),
                                    interval=3000,
                                    blit=True,
                                    repeat=True)
        plt.show()
