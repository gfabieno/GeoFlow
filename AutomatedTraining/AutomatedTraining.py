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
          use_tune: bool = False,
          **config):
    """
    Call `main` a succession of times as implied by `config`.

    :param main: A callable that oversees training and testing (i.e.
                 `..main.main`)
    :param args: Key-value pairs of argument names and values. `chain` will
                 fetch a different value at each iteration from values that are
                 tuples.

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
              logdir="logs",
              params=params,
              dataset=Dataset1Dsmall(),
              architecture=RCNN2D,
              ngpu=2)
    Output:
        A 3-step training with different losses.
    """
    params = deepcopy(params)
    for param_name, param_value in config:
        setattr(params, param_name, param_value)

    to_chain = {}
    for param_name, param_value in params.__dict__.items():
        if isinstance(param_value, tuple):
            to_chain[param_name] = param_value

    if to_chain:
        qties_segments = [len(param) for param in params.values()]
        qty_segments = qties_segments[0]
        assert all([qty == qty_segments for qty in qties_segments]), (
            "A hyperparameter sequence has a different length."
        )
    else:
        qty_segments = 1

    for segment in range(qty_segments):
        current_params = deepcopy(params)
        for param_name, param_value in to_chain:
            current_params[param_name] = param_value[segment]
        main(current_params, use_tune)


def optimize(architecture: RCNN2D.RCNN2D,
             params: Namespace,
             dataset: GeoDataset,
             logdir: str = "./logs",
             ngpu: int = 1,
             **config):
    """
    Call `chain` for all combinations of `chain`.

    :param args: Key-value pairs of argument names and values that will be
                 iterated upon.

    Sample usage:
        optimize(architecture=RCNN2D.RCNN2D,
                 params=Hyperparameters(),
                 dataset=Dataset1Dsmall(),
                 lr=[.0008, .0002])
    Output:
        Two calls to `chain` with different learning rates.
    """
    with ArchiveRepository() as archive:
        with archive.import_main() as main:
            logdir = archive.model
            tune.run(lambda config: chain(main, architecture, params, dataset,
                                          logdir, ngpu, use_tune=True,
                                          **config),
                     num_samples=1,
                     checkpoint_freq=1,
                     resources_per_trial={"gpu": ngpu},
                     config=config)


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
    args, config = parser.parse_known_args()

    args.architecture = getattr(RCNN2D, args.architecture)
    dataset_module = import_module("DefinedDataset." + args.dataset)
    args.dataset = getattr(dataset_module, args.dataset)()
    args.params = getattr(RCNN2D, args.params)()

    if args.debug:
        args.dataset.trainsize = 5
        args.dataset.validatesize = 0
        args.dataset.testsize = 5

    optimize(architecture=args.architecture,
             params=args.params,
             dataset=args.dataset,
             logdir=args.logdir,
             ngpu=args.ngpu,
             config=args.config)
