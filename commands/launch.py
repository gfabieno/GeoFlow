# -*- coding: utf-8 -*-
"""Launch custom training ."""

from argparse import Namespace
from itertools import product
from importlib import import_module

from archive import ArchiveRepository


def chain(main, **args):
    if "training" in args.keys():
        raise ValueError("Using `chain` implies training.")
    parameters = {key: value
                  for key, value in args.items()
                  if isinstance(value, list)}
    qty_segments = max([len(sequence) for sequence in parameters.values])
    constants = {key: [value]*qty_segments
                 for key, value in args.items()
                 if not isinstance(value, list)}
    parameters.update(constants)
    keys = parameters.keys()
    values = parameters.values()
    for current_values in zip(*values):
        current_parameters = {key: value for key, value
                              in zip(keys, current_values)}
        args = Namespace(training=2, **current_parameters)
        main(args)


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


args = dict(case="Case2Dtest_sourcedensity",
            lr=.0002,
            loss_ref=[[.5, .1, .1]],
            loss_vrms=[[.5, .6, .4]],
            loss_vint=[[.0, .3, .5]],
            loss_vdepth=[[.0, .0, .0]],
            epoch=[[100, 100, 50]],
            steps=20,
            batchsize=2,
            plot=0)
optimize(args)
