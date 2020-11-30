import os
import argparse

from GeoFlow.Trainer import Trainer
from GeoFlow.Tester import Tester


def main(args):
    logdir = args.logdir
    batch_size = args.batchsize
    dataset = args.dataset

    if args.debug == 1:
        dataset.trainsize = 5
        dataset.validatesize = 0
        dataset.testsize = 0

    # Generate the dataset.
    if args.training in [0, 2]:
        dataset.generate_dataset(ngpu=args.ngpu)

    if args.plot:
        dataset.animate()

    loss_scales = {'ref': args.loss_ref,
                   'vrms': args.loss_vrms,
                   'vint': args.loss_vint,
                   'vdepth': args.loss_vdepth}

    nn = RCNN2D(batch_size=batch_size,
                params=args.params,
                out_names=loss_scales.keys(),
                dataset=dataset)

    if args.training in [1, 2]:
        trainer = Trainer(nn=nn,
                          sequence=sequence,
                          checkpoint_dir=logdir,
                          learning_rate=args.lr,
                          beta_1=args.beta_1,
                          beta_2=args.beta_2,
                          epsilon=args.eps,
                          loss_scales=loss_scales,
                          use_weights=not args.no_weights)
        trainer.train_model(epochs=args.epochs,
                            initial_epoch=nn.current_epoch,
                            steps_per_epoch=args.steps)

    # Test model.
    if args.training == 3:
        tester = Tester(nn=nn, sequence=sequence, dataset=dataset)
        savepath = os.path.join(dataset.datatest, "pred")
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        tester.test_dataset(savepath=savepath)

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
    parser.add_argument("--params",
                        type=str,
                        default="Hyperparameters",
                        help="Name of hyperparameters from `RCNN2D` to use")
    parser.add_argument("--dataset",
                        type=str,
                        default="Dataset1Dsmall",
                        help="Name of dataset from `Datasets_define` to use")
    parser.add_argument("--logdir",
                        type=str,
                        default="./logs",
                        help="Directory in which to store the checkpoints")
    parser.add_argument("--training",
                        type=int,
                        default=0,
                        help="1: training only, 0: create dataset only, "
                        "2: training+dataset, 3: testing")
    parser.add_argument("--epochs",
                        type=int,
                        default=5,
                        help="number of epochs, with `steps` iterations per "
                             "epoch")
    parser.add_argument("--steps",
                        type=int,
                        default=100,
                        help="number of training iterations per epoch")
    parser.add_argument("--lr",
                        type=float,
                        default=0.0008,
                        help="learning rate")
    parser.add_argument("--beta_1",
                        type=float,
                        default=0.9,
                        help="beta1 for adadelta")
    parser.add_argument("--beta_2",
                        type=float,
                        default=0.98,
                        help="beta2 for adadelta")
    parser.add_argument("--eps",
                        type=float,
                        default=1e-5,
                        help="epsilon for adadelta")
    parser.add_argument("--batchsize",
                        type=int,
                        default=50,
                        help="size of the batches")
    parser.add_argument("--loss_ref",
                        type=float,
                        default=0.8,
                        help="weight of event referencing in loss")
    parser.add_argument("--loss_vrms",
                        type=float,
                        default=0.1,
                        help="weight of vrms in loss")
    parser.add_argument("--loss_vint",
                        type=float,
                        default=0.1,
                        help="weight of vint in loss")
    parser.add_argument("--loss_vdepth",
                        type=float,
                        default=0.0,
                        help="weight of vdepth in loss")
    parser.add_argument("--nmodel",
                        type=int,
                        default=1,
                        help="Number of models to train")
    parser.add_argument("--ngpu",
                        type=int,
                        default=1,
                        help="Number of gpu for data creation")
    parser.add_argument("--noise",
                        type=int,
                        default=0,
                        help="1: Add noise to the data")
    parser.add_argument("--plot",
                        type=int,
                        default=1,
                        help="1: Validate data by plotting.")
    parser.add_argument("--no_weights",
                        action='store_true',
                        help="Discard weighting in losses when training.")
    parser.add_argument("--restore_from",
                        type=str,
                        default=None,
                        help="The weights file used for inference. Defaults "
                             "to the last checkpoint in `args.logdir`.")
    parser.add_argument("--debug",
                        type=int,
                        default=00,
                        help="1: A small dataset of 5 examples is generate ")

    # Parse the input for training parameters.
    args = parser.parse_args()
    args.dataset = eval(args.dataset)()
    args.params = eval(args.params)()
    main(args)
