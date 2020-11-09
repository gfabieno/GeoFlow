# -*- coding: utf-8 -*-
"""Launch custom training."""

import sys
from os.path import realpath
from copy import deepcopy
from argparse import Namespace
from itertools import product
from importlib import import_module

import numpy as np

from archive import ArchiveRepository
from vrmslearn.RCNN2D import Hyperparameters
from Cases_define import *


def chain(main, **args):
    """Call `main` a succession of times as implied by `args`.

    :param main: a callable that oversees training and testing (i.e.
                 `..main.main`)
    :param args: Key-value pairs of argument names and values. `chain` will
                 fetch a different value at each iteration from values that are
                 tuples.
    """
    if "training" in args.keys():
        raise ValueError("Using `chain` implies training.")
    hyperparams = args["params"]
    if isinstance(hyperparams.freeze_to, tuple):
        args["params"] = generate_variations(hyperparams,
                                             freeze_to=hyperparams.freeze_to)
        args["params"] = tuple(args["params"])
    parameters = {key: value
                  for key, value in args.items()
                  if isinstance(value, tuple)}
    if parameters:
        qty_segments = max([len(sequence) for sequence in parameters.values()])
    else:
        qty_segments = 1
    constants = {key: [value]*qty_segments
                 for key, value in args.items()
                 if not isinstance(value, tuple)}
    parameters.update(constants)
    keys = parameters.keys()
    values = parameters.values()
    try:
        for current_values in zip(*values):
            current_parameters = {key: value for key, value
                                  in zip(keys, current_values)}
            args = Namespace(training=1, **current_parameters)
            main(args)
    except Exception as exception:
        print("Could not do or finish training:")
        print(exception)


def optimize(**args):
    """Call `chain` for all combinations of `args`.

    :param args: Key-value pairs of argument names and values. Values
                 that are lists will be iterated upon.
    """
    if "log_dir" in args.keys():
        raise ValueError("`optimize` manages checkpoint directories by "
                         "itself.")
    if "training" in args.keys():
        raise ValueError("Using `optimize` implies training.")
    parameters = {key: value
                  for key, value in args.items()
                  if isinstance(value, list)}
    constants = {key: [value]
                 for key, value in args.items()
                 if not isinstance(value, list)}
    parameters.update(constants)
    keys = parameters.keys()
    values = parameters.values()
    for current_values in product(*values):
        current_parameters = {key: value for key, value
                              in zip(keys, current_values)}
        with ArchiveRepository() as archive:
            archive.write(str(current_parameters))
            main = import_module("main").main
            chain(main, logdir=archive.model, **current_parameters)
            del sys.modules["main"]
            del main


def generate_variations(base_params, **variations):
    """Generate variations of an `Hyperparameters` object.

    :param base_params: A base `Hyperparameters` object.
    :param variations: Values with which to overwrite the attributes of
                       `base_params` with. The keys are attribute names and the
                       values will be iterated upon using a cartesian product.
    """
    hyperparams = []
    keys = variations.keys()
    values = variations.values()
    for current_values in product(*values):
        current_hyperparams = deepcopy(base_params)
        for key, value in zip(keys, current_values):
            setattr(current_hyperparams, key, value)
        hyperparams.append(current_hyperparams)
    return hyperparams


def drop_useless(hyperparams):
    """Drop useless hyperparameters combinations.

    Useless hyperparameters combinations have diltations in dimensions of
    length 1.

    :param hyperparams: A list of `Hyperparameters` objects.
    """
    to_keep = np.ones(len(hyperparams), dtype=bool)
    for i, p in enumerate(hyperparams):
        is_dilation_useful = True
        for key, value in p.__dict__.items():
            if value is None or "dilation" not in key:
                continue
            value = np.array(value)
            if (value != 1).any():
                attrib = key.split("_")[0]
                s_or_not = "s" if key[-1] == "s" else ""
                attrib = attrib + "_kernel" + s_or_not
                idx_dilation = np.nonzero(value != 1)
                idx_dilation = idx_dilation
                other_value = getattr(p, attrib)
                other_value = np.array(other_value)
                if not (other_value[idx_dilation] != 1).all():
                    is_dilation_useful = False
        to_keep[i] = is_dilation_useful
    hyperparams = np.array(hyperparams, dtype=object)
    hyperparams = hyperparams[to_keep]
    return list(hyperparams)


if __name__ == "__main__":
    hyperparams = Hyperparameters()
    hyperparams.freeze_to = (None, "ref", "vrms")
    all_hparams = [hyperparams]
    hyperparams = Hyperparameters()
    hyperparams.freeze_to = (None, "ref", "vrms")
    hyperparams.encoder_kernels = [[15, 1, 3],
                                   [1, 9, 3],
                                   [15, 1, 3],
                                   [1, 9, 3]]
    all_hparams.extend(generate_variations(hyperparams,
                                           encoder_dilations=[[[1, 1, 1],
                                                               [1, 1, 1],
                                                               [1, 1, 1],
                                                               [1, 1, 1]],
                                                              [[1, 1, 2],
                                                               [1, 1, 2],
                                                               [1, 1, 2],
                                                               [1, 1, 2]]],
                                           rcnn_kernel=[[15, 3, 1],
                                                        [15, 3, 3]],
                                           rcnn_dilation=[[1, 1, 1],
                                                          [1, 1, 2]]))

    hyperparams.use_cnn = True
    hyperparams.cnn_kernel = [1, 5]
    hyperparams.cnn_filters = 32
    hyperparams.cnn_dilation = [1, 2]
    all_hparams.append(deepcopy(hyperparams))
    all_hparams = drop_useless(all_hparams)

    checkpoint_1d = "logs/optimize-2D-kernels/a775455/2/model/0140.ckpt"
    checkpoint_1d = realpath(checkpoint_1d)
    optimize(params=all_hparams,
             case=Case2Dtest_complexity(),
             epochs=(80, 80, 50),
             steps=20,
             lr=.0002,
             beta_1=.9,
             beta_2=.98,
             eps=1e-5,
             batchsize=4,
             loss_ref=(.5, .0, .0),
             loss_vrms=(.5, .7, .0),
             loss_vint=(.0, .3, 1.),
             loss_vdepth=(.0, .0, .0),
             nmodel=1,
             ngpu=2,
             noise=0,
             plot=0,
             no_weights=False,
             restore_from=(checkpoint_1d, None, None))
