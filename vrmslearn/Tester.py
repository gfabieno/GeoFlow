# -*- coding: utf-8 -*-
"""
Test a neural network.
"""

import os
from os.path import join, basename
import fnmatch

import h5py as h5
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np

from vrmslearn.RCNN2D import RCNN2D
from vrmslearn.Case import Case
from vrmslearn.Sequence import Sequence


class Tester(object):
    """
    Test a neural network.
    """

    def __init__(self,
                 nn: RCNN2D,
                 sequence: Sequence,
                 case: Case):
        """
        Initialize the tester.

        :param nn: A Keras model.
        :type nn: RCNN2D
        :param sequence: A Keras `Sequence` object providing data.
        :type sequence: Sequence
        :param case: The current case describing the test dataset.
        :type case: Case
        """
        self.nn = nn
        self.sequence = sequence
        self.case = case

        self.out_names = self.nn.out_names

    def test_dataset(self,
                     savepath: str):
        """
        Evaluate and save predictions on all examples in `savepath`.

        The predictions are saved in hdf5 files.

        :param savepath: The path in which the test examples are found.
        """
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
        Get the labels and predictions for labels in `prednames`.

        :param prednames: List of name of the predictions in the example file
        :param savepath: The path in which the predictions are found
        :param examples: List of name of examples to get predictions of. If
                         None, predictions for all examples in savepath are
                         returned
        :param filename: The structure of the examples' filenames

        :return:
            labels: A dictionary of labels' name-values pairs.
            preds: DA dictionary of predictions' name-values pairs.
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
        Plot the labels and the predictions for each test sample.

        :param labelnames: List of names of the labels in the example file.
        :param savepath: The path in which the test examples are found.
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
                    axes[jj, 0].imshow(label[labelname],
                                       vmin=vmin,
                                       vmax=vmax,
                                       cmap='inferno',
                                       aspect='auto')
                    axes[jj, 1].imshow(pred[labelname],
                                       vmin=vmin,
                                       vmax=vmax,
                                       cmap='inferno',
                                       aspect='auto')

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
        Make an animation that shows the testing data, labels and predictions.

        :param labelnames: List of names of the labels to predict
        :param savepath: The path in which the test examples are found
        :param quantity: Number of examples to show. If None, show all examples
                         in the test set
        :param image: If True, labels and predictions are shown as images,
                      else plot 1D profiles.
        """
        if quantity is None:
            examples = [os.path.basename(f) for f in self.case.files["test"]
                        if os.path.basename(f) in os.listdir(savepath)]
        else:
            examples = [os.path.basename(self.case.files["test"][ii])
                        for ii in range(quantity)]

        labels, preds = self.get_preds(labelnames, savepath, examples=examples)
        datas = labels['input']
        datas = [np.reshape(el, [el.shape[0], -1]) for el in datas]

        if image:
            fig, axs = plt.subplots(3, 1 + len(labelnames), squeeze=False)
        else:
            fig, axs = plt.subplots(1, 1 + len(labelnames), squeeze=False)

        clip = 0.01
        vmax = np.max(datas) * clip
        vmin = -vmax
        im1 = axs[0, 0].imshow(datas[0],
                               animated=True,
                               vmin=vmin,
                               vmax=vmax,
                               aspect='auto',
                               cmap=plt.get_cmap('Greys'))
        axs[0, 0].set_title('data')
        ims = [im1]
        if image:
            src_pos, _ = self.case.acquire.set_rec_src()
            qty_shots = src_pos.shape[1]
            data = datas[0]
            data = data.reshape([data.shape[0], -1, qty_shots])
            im2 = axs[1, 0].imshow(data[..., qty_shots//2],
                                   animated=True,
                                   vmin=vmin,
                                   vmax=vmax,
                                   aspect='auto',
                                   cmap=plt.get_cmap('Greys'))
            axs[1, 0].set_title('center shot')
            ims.append(im2)

        labeld = {la: labels[la][0] for la in labelnames}
        predd = {la: preds[la][0] for la in labelnames}
        label, pred = self.case.label.postprocess(labeld, predd)

        for ii, labelname in enumerate(labelnames):
            if labelname == "ref":
                vmin, vmax = 0, 1
            else:
                vmin = self.case.model.vp_min
                vmax = self.case.model.vp_max
            y = np.arange(pred[labelname].shape[0])
            if image:
                im1 = axs[0, 1 + ii].imshow(label[labelname],
                                            vmin=vmin,
                                            vmax=vmax,
                                            animated=True,
                                            cmap='inferno',
                                            aspect='auto')
                im2 = axs[1, 1 + ii].imshow(pred[labelname],
                                            vmin=vmin,
                                            vmax=vmax,
                                            animated=True,
                                            cmap='inferno', aspect='auto')
                center_label = label[labelname][:, qty_shots//2]
                center_pred = pred[labelname][:, qty_shots//2]
                im3, = axs[2, 1 + ii].plot(center_label, y)
                im4, = axs[2, 1 + ii].plot(center_pred, y)
                axs[2, 1 + ii].set_ylim(np.min(y), np.max(y))
                axs[2, 1 + ii].set_xlim(vmin, vmax)
                axs[2, 1 + ii].invert_yaxis()
                axs[0, 1 + ii].set_title(labelname)
                ims.append(im1)
                ims.append(im2)
                ims.append(im3)
                ims.append(im4)
            else:
                im1, = axs[0, 1 + ii].plot(label[labelname][:, 0][:len(y)], y)
                im2, = axs[0, 1 + ii].plot(pred[labelname][:, 0][:len(y)], y)
                axs[0, 1 + ii].set_ylim(np.min(y), np.max(y))
                axs[0, 1 + ii].set_xlim(vmin, vmax)
                axs[0, 1 + ii].invert_yaxis()
                axs[0, 1 + ii].set_title(labelname)
                ims.append(im1)
                ims.append(im2)
        axs[2, 0].axis('off')
        plt.tight_layout()

        def animate(t):
            labeld = {la: labels[la][t] for la in labelnames}
            predd = {la: preds[la][t] for la in labelnames}
            label, pred = self.case.label.postprocess(labeld, predd)

            for ii, im in enumerate(ims):
                if ii == 0:
                    toplot = datas[t]
                    im.set_array(toplot)
                elif ii == 1 and image:
                    toplot = datas[t]
                    toplot = toplot.reshape([toplot.shape[0], -1, qty_shots])
                    im.set_array(toplot[..., qty_shots//2])
                else:
                    # Ignore the first column.
                    if image:
                        label_idx = (ii-2) // 4
                        item = (ii-2) % 4
                    else:
                        label_idx = (ii-1) // 2
                        item = (ii-1) % 2
                    if item == 0:
                        toplot = label[labelnames[label_idx]]
                    elif item == 1:
                        toplot = pred[labelnames[label_idx]]
                    elif item == 2:
                        toplot = label[labelnames[label_idx]][:, qty_shots//2]
                    elif item == 3:
                        toplot = pred[labelnames[label_idx]][:, qty_shots//2]
                    if image and item not in [2, 3]:
                        im.set_array(toplot)
                    else:
                        y = np.arange(toplot.shape[0])
                        im.set_data(toplot, y)
            return ims

        def init():
            return animate(0)

        _ = animation.FuncAnimation(fig,
                                    animate,
                                    init_func=init,
                                    frames=len(datas),
                                    interval=3000,
                                    blit=True,
                                    repeat=True)
        plt.show()
