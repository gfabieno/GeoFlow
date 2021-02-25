# -*- coding: utf-8 -*-
"""
Launch dataset generation, training or testing.
"""

import argparse


def main(args, use_tune=False):
    dataset = args.dataset
    if isinstance(args.gpus, int):
        args.gpus = list(range(args.gpus))

    if args.debug:
        dataset.trainsize = 4
        dataset.validatesize = 0
        dataset.testsize = 4

        args.params.batch_size = 2
        args.params.epochs = 2
        args.params.steps_per_epoch = 1

    if args.plot:
        dataset.acquire.plot_acquisition_geometry()

    if args.training in [0, 2]:
        dataset.generate_dataset(ngpu=args.ngpu)

    if args.plot:
        dataset.animate()

    if args.training != 0:
        phase = "train" if args.training in [1, 2] else "test"
        inputs, _, _, _ = dataset.get_example(toinputs=args.nn.toinputs)
        input_shapes = {name: input.shape for name, input in inputs.items()}
        nn = args.nn(dataset=dataset,
                     input_shapes=input_shapes,
                     params=args.params,
                     checkpoint_dir=args.logdir,
                     devices=args.gpus,
                     run_eagerly=args.eager)
        tfdataset = dataset.tfdataset(phase=phase,
                                      tooutputs=nn.tooutputs,
                                      toinputs=nn.toinputs,
                                      batch_size=args.params.batch_size)

        # Train model.
        if args.training in [1, 2]:
            nn.launch_training(tfdataset, use_tune)

        # Test model.
        if args.training == 3:
            nn.launch_testing(tfdataset, args.savedir)
            if args.plot:
                pred_dir = args.savedir or type(nn).__name__
                dataset.animate(phase='test', plot_preds=True,
                                pred_dir=pred_dir)


def int_or_list(arg):
    if arg is None:
        return None
    try:
        arg = int(arg)
    except ValueError:
        arg = list(arg)
    return arg


if __name__ == "__main__":
    from importlib import import_module

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
    parser.add_argument("--gpus",
                        type=int_or_list,
                        default=None,
                        help="Either the quantity of GPUs or a list of GPU "
                             "IDs to use in data creation, training and "
                             "inference. Use a string representation for "
                             "lists of GPU IDs, e.g. `'[0, 1]'`. By default, "
                             "use all available GPUs.")
    parser.add_argument("--savedir",
                        type=str,
                        default=None,
                        help="The name of the subdirectory within the dataset "
                             "test directory to save predictions in. Defaults "
                             "to the name of the network class.")
    parser.add_argument("--plot",
                        action='store_true',
                        help="Validate data by plotting.")
    parser.add_argument("--debug",
                        action='store_true',
                        help="Generate a small dataset of 5 examples.")
    parser.add_argument("--eager",
                        action='store_true',
                        help="Run the Keras model eagerly, for debugging.")

    args, unknown_args = parser.parse_known_args()
    nn_module = import_module("DefinedNN." + args.nn)
    args.nn = getattr(nn_module, args.nn)
    is_training = args.training in [1, 2]
    args.params = getattr(nn_module, args.params)(is_training=is_training)
    dataset_module = import_module("DefinedDataset." + args.dataset)
    args.dataset = getattr(dataset_module, args.dataset)()
    for arg, value in zip(unknown_args[::2], unknown_args[1::2]):
        arg = arg.strip('-')
        if arg in args.params.__dict__.keys():
            setattr(args.params, arg, value)
        else:
            raise ValueError(
                f"Argument `{arg}` not recognized. Could not match it to an "
                f"existing hyerparameter."
            )
    main(args)
