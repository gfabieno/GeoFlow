# -*- coding: utf-8 -*-
"""Launch custom training ."""

import sys
from os.path import realpath
from copy import deepcopy
from argparse import Namespace
from itertools import product
from importlib import import_module

from archive import ArchiveRepository
from vrmslearn.RCNN2D import Hyperparameters
from Cases_define import *


def chain(main, **args):
    if "training" in args.keys():
        raise ValueError("Using `chain` implies training.")
    hyperparams = args["params"]
    if isinstance(hyperparams.freeze_to, tuple):
        args["params"] = generate_variations(hyperparams,
                                             freeze_to=hyperparams.freeze_to)
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
            args = Namespace(training=2, **current_parameters)
            main(args)
    except Exception as exception:
        print("Could not do or finish training:")
        print(exception)


def optimize(**args):
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
    hyperparams = []
    keys = variations.keys()
    values = variations.values()
    for current_values in product(*values):
        current_hyperparams = deepcopy(base_params)
        for key, value in zip(keys, current_values):
            setattr(current_hyperparams, key, value)
        hyperparams.append(current_hyperparams)
    return hyperparams


hyperparams = generate_variations(Hyperparameters(),
                                  freeze_to=[None, (None, "ref", "vrms")],
                                  encoder_kernels=[[[15, 1, 1],
                                                    [1, 9, 1],
                                                    [15, 1, 1],
                                                    [1, 9, 1]],
                                                   [[15, 1, 3],
                                                    [1, 9, 3],
                                                    [15, 1, 3],
                                                    [1, 9, 3]]],
                                  encoder_dilations=[[[1, 1, 1],
                                                      [1, 1, 1],
                                                      [1, 1, 1],
                                                      [1, 1, 1]],
                                                     [[1, 1, 2],
                                                      [1, 1, 2],
                                                      [1, 1, 2],
                                                      [1, 1, 2]]],
                                  rcnn_kernel=[[15, 3, 1], [15, 3, 3]],
                                  rcnn_dilation=[[1, 1, 1], [1, 1, 2]],
                                  decode_kernel=[[1, 1], [1, 5]])
checkpoint_1d = "logs/optimize-2D-kernels/a775455/2/model/0140.ckpt"
checkpoint_1d = realpath(checkpoint_1d)
optimize(params=hyperparams,
         case=Case2Dtest_complexity(),
         epochs=(100, 100, 50),
         steps=20,
         lr=[.0002, .00008],
         beta_1=.9,
         beta_2=.98,
         eps=1e-5,
         batchsize=8,
         loss_ref=(.5, .0, .0),
         loss_vrms=(.5, .7, .0),
         loss_vint=(.0, .3, 1.),
         loss_vdepth=(.0, .0, .0),
         nmodel=1,
         ngpu=2,
         noise=0,
         plot=0,
         no_weights=False,
         restore_from=[None, (checkpoint_1d, None, None)])
