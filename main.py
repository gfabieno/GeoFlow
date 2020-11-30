import os
import argparse

from GeoFlow.Trainer import Trainer
from GeoFlow.Tester import Tester


def main(args):
    logdir = args.logdir
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

    nn = RCNN2D(batch_size=batch_size,
                params=args.params,
                out_names=loss_scales.keys(),
                dataset=dataset)

    if args.training in [1, 2]:
        trainer = Trainer(nn=nn,
                          sequence=sequence,
                          checkpoint_dir=logdir)
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
                        help="Name of dataset from `DefinedDataset` to use")
    parser.add_argument("--logdir",
                        type=str,
                        default="./logs",
                        help="Directory in which to store the checkpoints")
    parser.add_argument("--training",
                        type=int,
                        default=0,
                        help="1: training only, 0: create dataset only, "
                        "2: training+dataset, 3: testing")
    parser.add_argument("--ngpu",
                        type=int,
                        default=1,
                        help="Number of gpu for data creation")
    parser.add_argument("--plot",
                        type=int,
                        default=1,
                        help="1: Validate data by plotting.")
                        action='store_true',
    parser.add_argument("--debug",
                        type=int,
                        default=00,
                        help="1: A small dataset of 5 examples is generate ")

    args = parser.parse_args()
    args.dataset = eval(args.dataset)()
    args.params = eval(args.params)()
    main(args)
