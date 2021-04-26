# -*- coding: utf-8 -*-
"""
Launch hyperoptimization and chain training stages.

This module allows chaining multiple calls to a `main` script such as
`..main.main` through `chain` and launching `chain` with different combinations
of hyperparameters through `optimize`. To use different combinations of
architecture hyperparameters (`GeoFlow.DefinedNN.RCNN2D.Hyperparameters`) in
launching a main script, the alterations to the base hyperparameters must be
fed to `optimize` as key-value pairs of hyperparameter names and lists possible
values. `optimize` processes all combinations of items from arguments that are
lists. This module leverages `AutomatedTraining.Archive` to make sure
modifications in the repository during training do not impact an ongoing
training. `optimize` automatically fetches the archived main script.
"""

from os.path import split
from copy import deepcopy
from argparse import Namespace
from typing import Callable

from ray import tune
from tensorflow.config import list_physical_devices

from .Archive import ArchiveRepository


def chain(main: Callable,
          args: Namespace,
          use_tune: bool = False,
          **config):
    """
    Call `main` a succession of times as implied by `config`.

    :param main: A callable that oversees training and testing (i.e.
                 `..main.main`)
    :param args: Parsed arguments.
    :param config: Key-value pairs of argument names and values. `chain` will
                   fetch a different value at each iteration from values that
                   are tuples in `config` and `args.params`.

    Sample usage:
        from GeoFlow.__main__ import main, parse_args
        from GeoFlow.AutomatedTraining import chain

        args = parse_args()
        args.params.loss_scales = ({'ref': .5, 'vrms': .5,
                                    'vint': .0, 'vdepth': .0},
                                   {'ref': .0, 'vrms': .7,
                                    'vint': .3, 'vdepth': .0},
                                   {'ref': .0, 'vrms': .0,
                                    'vint': 1., 'vdepth': .0})
        chain(main, args)

    Output:
        A 3-step training with different losses.
    """
    args.train = True
    args.infer = False
    args.plot = False
    params = deepcopy(args.params)
    for param_name, param_value in config.items():
        setattr(params, param_name, param_value)

    to_chain = {}
    for param_name, param_value in params.__dict__.items():
        if isinstance(param_value, tuple):
            to_chain[param_name] = param_value

    if to_chain:
        qties_segments = [len(param) for param in to_chain.values()]
        qty_segments = qties_segments[0]
        is_equal_length = all([qty == qty_segments for qty in qties_segments])
        assert is_equal_length, ("Some hyperparameter sequence has a "
                                 "different length.")
    else:
        qty_segments = 1

    for segment in range(qty_segments):
        current_params = deepcopy(params)
        for param_name, param_value in to_chain.items():
            setattr(current_params, param_name, param_value[segment])
        if use_tune:
            with tune.checkpoint_dir(step=1) as checkpoint_dir:
                logdir, _ = split(checkpoint_dir)
        args.params = current_params
        main(args, use_tune)


def optimize(args: Namespace, **config):
    """
    Call `chain` for all combinations of `args.params`.

    :param args: Parsed arguments.
    :param config: Key-value pairs of argument names and values that will be
                   iterated upon.

    Sample usage:
        from GeoFlow.__main__ import parse_args
        from GeoFlow.AutomatedTraining import optimize

        args = parse_args()
        args.params.lr = [.0008, .0002]
        optimize(args)

    Output:
        Two calls to `chain` with different learning rates.
    """
    with ArchiveRepository(args.logdir) as archive:
        with archive.import_main() as main:
            args.logdir = archive.model

            grid_search_config = deepcopy(config)
            for key, value in config.items():
                if isinstance(value, list):
                    value = tune.grid_search(value)
                grid_search_config[key] = value
            if args.gpus is None:
                ngpu = len(list_physical_devices('GPU'))
            elif isinstance(args.gpus, list):
                ngpu = len(list)
            else:
                ngpu = args.gpus
            tune.run(lambda config: chain(main, args, use_tune=True, **config),
                     num_samples=1,
                     local_dir=args.logdir,
                     resources_per_trial={"gpu": ngpu},
                     config=grid_search_config)
