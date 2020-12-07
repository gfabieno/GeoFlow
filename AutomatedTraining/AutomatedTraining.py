# -*- coding: utf-8 -*-
"""
Launch hyperoptimization and chain training stages.

This module allows chaining multiple calls to a `main` script such as
`..main.main` through `chain` and lauching `chain` with different combinations
of hyperparameters through `optimize`. To use different combinations of
architecture hyperparameters (`GeoFlow.RCNN2D.Hyperparameters`) in launching
a main script, the alterations to the base hyperparameters must be fed to
`optimize` as key-value pairs of hyperparameter names and lists possible
values. `optimize` processes all combinations of items from arguments that are
lists. This module leverages `automated_training.archive` to make sure
modifications in the repository during training do not impact an ongoing
training. `optimize` automatically fetches the archived main script.
"""

from copy import deepcopy
from argparse import Namespace, ArgumentParser
from importlib import import_module
from typing import Callable

from ray import tune

from .Archive import ArchiveRepository
from GeoFlow import RCNN2D
from GeoFlow.GeoDataset import GeoDataset


def chain(main: Callable,
          architecture: RCNN2D.RCNN2D,
          params: Namespace,
          dataset: GeoDataset,
          logdir: str = "./logs",
          ngpu: int = 1,
          debug: bool = False,
          eager: bool = False,
          use_tune: bool = False,
          **config):
    """
    Call `main` a succession of times as implied by `config`.

    :param main: A callable that oversees training and testing (i.e.
                 `..main.main`)
    :param architecture: Name of the architecture from `RCNN2D` to use.
    :param params: Name of hyperparameters from `RCNN2D` to use.
    :param dataset: Name of dataset from `DefinedDataset` to use.
    :param logdir: Directory in which to store the checkpoints.
    :param ngpu: Quantity of GPUs for data creation.
    :param debug: Generate a small dataset of 5 examples.
    :param eager: Run the Keras model eagerly, for debugging.
    :param use_tune: Whether to use `ray[tune]` or not.
    :param config: Key-value pairs of argument names and values. `chain` will
                   fetch a different value at each iteration from values that
                   are tuples.

    Sample usage:
        from main import main
        params = Hyperparameters()
        params.loss_scales = ({'ref': .5, 'vrms': .5,
                               'vint': .0, 'vdepth': .0},
                              {'ref': .0, 'vrms': .7,
                               'vint': .3, 'vdepth': .0},
                              {'ref': .0, 'vrms': .0,
                               'vint': 1., 'vdepth': .0})
        chain(main,
              architecture=RCNN2D,
              params=params,
              dataset=Dataset1Dsmall(),
              logdir="logs",
              ngpu=2)
    Output:
        A 3-step training with different losses.
    """
    params = deepcopy(params)
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
            if segment > 0:
                with tune.checkpoint_dir(step=segment-1) as checkpoint_dir:
                    logdir = checkpoint_dir
        args = Namespace(architecture=architecture, params=current_params,
                         dataset=dataset, logdir=logdir, training=1, ngpu=ngpu,
                         plot=False, debug=debug, eager=eager)
        main(args, use_tune)


def optimize(architecture: RCNN2D.RCNN2D,
             params: Namespace,
             dataset: GeoDataset,
             logdir: str = "./logs",
             ngpu: int = 1,
             debug: bool = False,
             eager: bool = False,
             **config):
    """
    Call `chain` for all combinations of `config`.

    :param architecture: Name of the architecture from `RCNN2D` to use.
    :param params: Name of hyperparameters from `RCNN2D` to use.
    :param dataset: Name of dataset from `DefinedDataset` to use.
    :param logdir: Directory in which to store the checkpoints.
    :param ngpu: Quantity of GPUs for data creation.
    :param debug: Generate a small dataset of 5 examples.
    :param eager: Run the Keras model eagerly, for debugging.
    :param config: Key-value pairs of argument names and values that will be
                   iterated upon.

    Sample usage:
        optimize(architecture=RCNN2D.RCNN2D,
                 params=Hyperparameters(),
                 dataset=Dataset1Dsmall(),
                 lr=[.0008, .0002])
    Output:
        Two calls to `chain` with different learning rates.
    """
    with ArchiveRepository(logdir) as archive:
        with archive.import_main() as main:
            logdir = archive.model

            grid_search_config = deepcopy(config)
            for key, value in config.items():
                if isinstance(value, list):
                    value = tune.grid_search(value)
                grid_search_config[key] = value
            tune.run(lambda config: chain(main, architecture, params, dataset,
                                          logdir, ngpu, debug, eager,
                                          use_tune=True, **config),
                     num_samples=1,
                     local_dir=logdir,
                     resources_per_trial={"gpu": ngpu},
                     config=grid_search_config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--architecture",
                        type=str,
                        default="RCNN2D",
                        help="Name of the architecture from `RCNN2D` to use.")
    parser.add_argument("--params",
                        type=str,
                        default="Hyperparameters",
                        help="Name of hyperparameters from `RCNN2D` to use.")
    parser.add_argument("--dataset",
                        type=str,
                        default="Dataset1Dsmall",
                        help="Name of dataset from `DefinedDataset` to use.")
    parser.add_argument("--logdir",
                        type=str,
                        default="./logs",
                        help="Directory in which to store the checkpoints.")
    parser.add_argument("--ngpu",
                        type=int,
                        default=1,
                        help="Quantity of GPUs for data creation.")
    parser.add_argument("--debug",
                        action='store_true',
                        help="Generate a small dataset of 5 examples.")
    parser.add_argument("--eager",
                        action='store_true',
                        help="Run the Keras model eagerly, for debugging.")
    args, config = parser.parse_known_args()
    config = {name[2:]: eval(value) for name, value
              in zip(config[::2], config[1::2])}
    args.architecture = getattr(RCNN2D, args.architecture)
    dataset_module = import_module("DefinedDataset." + args.dataset)
    args.dataset = getattr(dataset_module, args.dataset)()
    args.params = getattr(RCNN2D, args.params)()

    if args.debug:
        config["epochs"] = 1
        config["steps_per_epoch"] = 5

    optimize(architecture=args.architecture,
             params=args.params,
             dataset=args.dataset,
             logdir=args.logdir,
             ngpu=args.ngpu,
             debug=args.debug,
             eager=args.eager,
             **config)
