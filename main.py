# -*- coding: utf-8 -*-
"""
Launch dataset generation, training or testing.
"""

import argparse


def main(args, use_tune=False):
    dataset = args.dataset

    if args.debug:
        dataset.trainsize = 5
        dataset.validatesize = 0
        dataset.testsize = 5

        args.params.batch_size = 2

    # Generate the dataset.
    if args.training in [0, 2]:
        dataset.generate_dataset(ngpu=args.ngpu)

    if args.plot:
        dataset.animate()

    if args.training != 0:
        phase = "train" if args.training in [1, 2] else "test"
        nn = args.nn(dataset=dataset,
                     phase=phase,
                     params=args.params,
                     checkpoint_dir=args.logdir)

        # Train model.
        if args.training in [1, 2]:
            nn.setup_training(run_eagerly=args.eager)
            nn.launch_training(use_tune)

        # Test model.
        if args.training == 3:
            nn.launch_testing()
            if args.plot:
                nn_name = type(nn).__name__
                dataset.animate(phase='test', plot_preds=True, nn_name=nn_name)


if __name__ == "__main__":
    from importlib import import_module
    from GeoFlow import RCNN2D

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to parse for training
    parser.add_argument("--nn",
                        type=str,
                        default="RCNN2D",
                        help="Name of the neural net from `RCNN2D` to use.")
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
    parser.add_argument("--eager",
                        action='store_true',
                        help="Run the Keras model eagerly, for debugging.")

    args = parser.parse_args()
    args.nn = getattr(RCNN2D, args.nn)
    dataset_module = import_module("DefinedDataset." + args.dataset)
    args.dataset = getattr(dataset_module, args.dataset)()
    args.params = getattr(RCNN2D, args.params)()
    main(args)
