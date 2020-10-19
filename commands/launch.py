# -*- coding: utf-8 -*-
"""Launch custom training ."""

from argparse import Namespace
from itertools import product
from importlib import import_module

from archive import chdir, archive_current_state

log_dir, model_dir, code_dir = archive_current_state()
chdir(code_dir)


def chain(main, **args):
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
        args = Namespace(**current_parameters)
        main(args)


def optimize(**args):
    if "log_dir" in args.keys():
        raise ValueError("`optimize` manages checkpoint directories by "
                         "itself.")
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
        log_dir, _, code_dir = archive_current_state()
        chdir(code_dir)
        main = import_module("main").main
        chain(main, **current_parameters)
        chdir("../../../../..")


args = dict(logdir=model_dir,
            case="Case2Dtest_sourcedensity",
            training=1,
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
