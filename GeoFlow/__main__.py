# -*- coding: utf-8 -*-
"""
Launch dataset generation, training or testing.
"""

import argparse


def main(args=None, use_tune=False):
    if args is None:
        args = parse_args()

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

    if args.generate:
        if args.plot:
            dataset.acquire.plot_acquisition_geometry()
        dataset.generate_dataset(gpus=args.gpus)
        if args.plot:
            dataset.animate()

    if args.train or args.test:
        inputs, _, _, _ = dataset.get_example(toinputs=args.nn.toinputs)
        input_shapes = {name: input.shape for name, input in inputs.items()}
        nn = args.nn(dataset=dataset,
                     input_shapes=input_shapes,
                     params=args.params,
                     checkpoint_dir=args.logdir,
                     devices=args.gpus,
                     run_eagerly=args.eager)
        if args.train:
            tfdataset = dataset.tfdataset(phase="train",
                                          tooutputs=nn.tooutputs,
                                          toinputs=nn.toinputs,
                                          batch_size=args.params.batch_size)
            tfvalidate = dataset.tfdataset(phase="validate",
                                           tooutputs=nn.tooutputs,
                                           toinputs=nn.toinputs,
                                           batch_size=args.params.batch_size)
            nn.launch_training(tfdataset, tfvalidate=tfvalidate,
                               use_tune=use_tune)
        if args.test:
            tfdataset = dataset.tfdataset(phase="test",
                                          tooutputs=nn.tooutputs,
                                          toinputs=nn.toinputs,
                                          batch_size=args.params.batch_size)
            nn.launch_testing(tfdataset, args.savedir)
            if args.plot:
                pred_dir = args.savedir or type(nn).__name__
                dataset.animate(phase='test', plot_preds=True,
                                pred_dir=pred_dir)


def parse_args():
    from GeoFlow import DefinedDataset, DefinedNN

    args, unknown_args = parser.parse_known_args()
    args.nn = getattr(DefinedNN, args.nn, args.nn)
    is_training = args.training in [1, 2]
    args.params = getattr(DefinedNN, args.nn, args.params)
    args.params = args.params(is_training=is_training)
    args.dataset = getattr(DefinedDataset, args.dataset)()
    for arg, value in zip(unknown_args[::2], unknown_args[1::2]):
        arg = arg.strip('-')
        if arg in args.params.__dict__.keys():
            setattr(args.params, arg, eval(value))
        else:
            raise ValueError(f"Argument `{arg}` not recognized. Could not "
                             f"match it with an existing hyperparameter.")
    return args


def int_or_list(arg):
    if arg is None:
        return None
    arg = eval(arg)
    assert isinstance(arg, (int, list))
    return arg


parser = argparse.ArgumentParser()
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
parser.add_argument("--generate",
                    action='store_true',
                    help="Launch dataset generation.")
parser.add_argument("--train",
                    action='store_true',
                    help="Launch training.")
parser.add_argument("--test",
                    action='store_true',
                    help="Launch testing.")
parser.add_argument("--gpus",
                    type=int_or_list,
                    default=None,
                    help="Either the quantity of GPUs or a list of GPU "
                         "IDs to use in data creation, training and "
                         "inference. Use a string representation for "
                         "lists of GPU IDs, e.g. `'[0, 1]'` or `[0,1]`. "
                         "By default, use all available GPUs.")
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


if __name__ == "__main__":
    main()
