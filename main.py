import os
import re
import argparse

from Cases_define import *
from vrmslearn.RCNN2D import RCNN2D
from vrmslearn.Trainer import Trainer
from vrmslearn.Tester import Tester
from vrmslearn.Sequence import Sequence


def main(args):
    logdir = args.logdir
    batch_size = args.batchsize

    # Define the parameters.
    case = eval(args.case)()

    # Generate the dataset.
    if args.training in [0, 2]:
        case.generate_dataset(ngpu=args.ngpu)

    if args.plot:
        case.animated_dataset()

    loss_scales = {'ref': args.loss_ref,
                   'vrms': args.loss_vrms,
                   'vint': args.loss_vint,
                   'vdepth': args.loss_vdepth}

    sizes = case.get_dimensions()
    if args.restore_from is None:
        restore_from = find_latest_checkpoint(logdir)
    else:
        restore_from = args.restore_from
    if restore_from is not None:
        filename = os.path.split(restore_from)[-1]
        current_epoch = int(filename[:4])
    else:
        current_epoch = 0
    nn = RCNN2D(input_size=sizes[0],
                batch_size=batch_size,
                alpha=0.1,
                beta=0.1,
                use_peepholes=args.use_peepholes,
                out_names=loss_scales.keys(),
                restore_from=restore_from,
                freeze_to=args.freeze_to,
                case=case)

    # Train the model.
    if args.training in [1, 2]:
        sequence = Sequence(is_training=True,
                            case=case,
                            batch_size=batch_size,
                            input_size=sizes[0],
                            depth_size=sizes[-1][0],
                            out_names=loss_scales.keys())
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
                            initial_epoch=current_epoch,
                            steps_per_epoch=args.steps)

    # Test model.
    if args.training == 3:
        sequence = Sequence(is_training=False,
                            case=case,
                            batch_size=batch_size,
                            input_size=sizes[0],
                            depth_size=sizes[-1][0],
                            out_names=loss_scales.keys())
        tester = Tester(nn=nn, sequence=sequence, case=case)
        savepath = os.path.join(case.datatest, "pred")
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        tester.test_dataset(savepath=savepath)

        if args.plot:
            tester.animated_predictions(labelnames=['ref', 'vrms',
                                                    'vint', 'vdepth'],
                                        savepath=savepath,
                                        image="2D" in args.case)


def find_latest_checkpoint(logdir):
    expr = re.compile(r"[0-9]{4}\.ckpt")
    checkpoints = [f for f in os.listdir(logdir) if expr.match(f)]
    if checkpoints:
        checkpoints.sort()
        restore_from = checkpoints[-1]
        restore_from = os.path.join(logdir, restore_from)
    else:
        restore_from = None
    return restore_from


if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to parse for training
    parser.add_argument("--case",
                        type=str,
                        default="Case1Dsmall",
                        help="Name of the case from `Cases_define` to use")
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
    parser.add_argument("--use_peepholes",
                        type=int,
                        default=0,
                        help="1: Use peep hole in LSTM")
    parser.add_argument("--plot",
                        type=int,
                        default=1,
                        help="1: Validate data by plotting.")
    parser.add_argument("--no_weights",
                        action='store_false',
                        help="Discard weighting in losses when training.")
    parser.add_argument("--restore_from",
                        type=str,
                        default=None,
                        help="The weights file used for inference. Defaults "
                             "to the last checkpoint in `args.logdir`.")
    parser.add_argument("--freeze_to",
                        type=str,
                        default=None,
                        help="A label name. All layers before this label's "
                             "decoder will not be trainable.")

    # Parse the input for training parameters.
    args, unparsed = parser.parse_known_args()
    main(args)
