# -*- coding: utf-8 -*-
"""
Launch hyperoptimization and chain training stages.

This module allows chaining multiple calls to a `main` script such as
`..main.main` through `chain` and lauching `chain` with different combinations
of hyperparameters through `optimize`. To use different combinations of
architecture hyperparameters (`GeoFlow.RCNN2D.Hyperparameters`) in launching
a main script, the combinations must be placed in a list beforehand through
`generate_variations`. `optimize` processes all combinations of items from
arguments that are lists. This module leverages `automated_training.archive`
to make sure modifications in the repository during training do not impact an
ongoing training. `optimize` automatically fetches the archived main script.
"""

import sys
from copy import deepcopy
from argparse import Namespace
from itertools import product
from importlib import import_module

import numpy as np

from archive import ArchiveRepository
from GeoFlow.RCNN2D import Hyperparameters


def chain(main, **args):
    """
    Call `main` a succession of times as implied by `args`.

    :param main: A callable that oversees training and testing (i.e.
                 `..main.main`)
    :param args: Key-value pairs of argument names and values. `chain` will
                 fetch a different value at each iteration from values that are
                 tuples.

    Sample usage:
        from main import main
        chain(main,
              logdir="logs",
              params=Hyperparameters(),
              case=Case1Dsmall(),
              epochs=(100, 100, 50),
              steps=20,
              lr=.0008,
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
              restore_from=None)
    Output:
        A 3-step training with different quantities of epochs and losses.
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
    """
    Call `chain` for all combinations of `args`.

    :param args: Key-value pairs of argument names and values. Values
                 that are lists will be iterated upon.

    Sample usage:
        optimize(params=Hyperparameters(),
                 case=Case1Dsmall(),
                 epochs=(100, 100, 50),
                 steps=20,
                 lr=[.0008, .0002],
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
                 restore_from=None)
    Output:
        Two calls to `chain` with different learning rates.
    """
    if "logdir" in args.keys():
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
    """
    Generate variations of an `Hyperparameters` object.

    :param base_params: A base `Hyperparameters` object.
    :param variations: Values with which to overwrite the attributes of
                       `base_params` with. The keys are attribute names and the
                       values will be iterated upon using a cartesian product.

    Sample usage:
        hyperparams = Hyperparameters()
        generate_variations(hyperparams, rcnn_kernel=[[15, 3, 1],
                                                      [15, 3, 3]])
    Output:
        A list of two `Hyperparameters` objects with different `rcnn_kernel`
        attributes.
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
    """
    Drop useless hyperparameters combinations.

    Useless hyperparameters combinations have diltations in dimensions of
    length 1.

    :param hyperparams: A list of `Hyperparameters` objects.

    Sample usage:
        hyperparams = Hyperparameters()
        hyperparams.rcnn_kernel = [15, 3, 1]
        hyperparams.dilation = [1, 1, 2]
        drop_useless([hyperparams])
    Output:
        An empty list.
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
    from Cases_define import *

    optimize(params=Hyperparameters(),
             case=Case1Dsmall(),
             epochs=(100, 100, 50),
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
             restore_from=None)
