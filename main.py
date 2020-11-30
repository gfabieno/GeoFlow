# -*- coding: utf-8 -*-
"""
Launch dataset generation, training or testing.
"""

import argparse


def main(args):
    dataset = args.dataset

    if args.debug:
        dataset.trainsize = 5
        dataset.validatesize = 0
        dataset.testsize = 0

    # Generate the dataset.
    if args.training in [0, 2]:
        dataset.generate_dataset(ngpu=args.ngpu)

    if args.plot:
        dataset.animate()

    architecture = args.architecture(dataset=dataset,
                                     params=args.params,
                                     checkpoint_dir=args.logdir)

    if args.training in [1, 2]:
        architecture.launch_training()

    # Test model.
    if args.training == 3:
        architecture.launch_test()
        if args.plot:
            is_2d = sizes[0][2] != 1
            tester.animated_predictions(labelnames=['ref', 'vrms',
                                                    'vint', 'vdepth'],
                                        savepath=savepath,
                                        image=is_2d)


if __name__ == "__main__":
    from GeoFlow.RCNN2D import *
    from DefinedDataset import *

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to parse for training
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
    parser.add_argument("--training",
                        type=int,
                        default=0,
                        help="0: create dataset only; 1: training only; "
                        "2: training+dataset; 3: testing.")
    parser.add_argument("--ngpu",
                        type=int,
                        default=1,
                        help="Quantity of GPUs for data creation.")
    parser.add_argument("--plot",
                        action='store_true',
                        help="Validate data by plotting.")
    parser.add_argument("--debug",
                        action='store_true',
                        help="Generate a small dataset of 5 examples.")

    args = parser.parse_args()
    args.architecture = eval(args.architecture)()
    args.dataset = eval(args.dataset)()
    args.params = eval(args.params)()
    main(args)
