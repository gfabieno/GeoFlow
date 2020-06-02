from Cases_define import *
from vrmslearn.RCNN2D import RCNN2D
from vrmslearn.Trainer import Trainer
from vrmslearn.Tester import Tester
import argparse
import tensorflow as tf
import os


if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to parse for training
    parser.add_argument(
        "--case",
        type=str,
        help="Name of the case class from `Cases_define` to use"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./logs",
        help="Directory in which to store the checkpoints"
    )
    parser.add_argument(
        "--training",
        type=int,
        default=0,
        help="1: training only, 0: create dataset only, 2: training+dataset, "
             "3: testing"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0008,
        help="learning rate "
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="epsilon for adadelta"
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=50,
        help="size of the batches"
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="beta1 for adadelta"
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.98,
        help="beta2 for adadelta"
    )
    parser.add_argument(
        "--loss_ref",
        type=float,
        default=0.8,
        help="weight of event referencing in loss"
    )
    parser.add_argument(
        "--loss_vrms",
        type=float,
        default=0.1,
        help="weight of vrms in loss"
    )
    parser.add_argument(
        "--loss_vint",
        type=float,
        default=0.1,
        help="weight of vint in loss"
    )
    parser.add_argument(
        "--loss_vdepth",
        type=float,
        default=0.0,
        help="weight of vdepth in loss"
    )
    parser.add_argument(
        "--nmodel",
        type=int,
        default=1,
        help="Number of models to train"
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=1,
        help="Number of gpu for data creation"
    )
    parser.add_argument(
        "--noise",
        type=int,
        default=0,
        help="1: Add noise to the data"
    )
    parser.add_argument(
        "--use_peepholes",
        type=int,
        default=0,
        help="1: Use peep hole in LSTM"
    )
    parser.add_argument(
        "--plot",
        type=int,
        default=1,
        help="1: Validate data by plotting."
    )

    # Parse the input for training parameters.
    args, unparsed = parser.parse_known_args()

    logdir = args.logdir
    batch_size = args.batchsize

    # Define the parameters.
    case = eval(args.case)(
        trainsize=10000,
        validatesize=0,
        testsize=100,
    )
    if args.training == 3 and case.testsize < batch_size:
        batch_size = case.testsize

    # Generate the dataset.
    if args.training in [0, 2]:
        case.generate_dataset(ngpu=args.ngpu)

    if args.plot:
        case.animated_dataset()

    sizes = case.get_dimensions()
    nn = RCNN2D(
        input_size=sizes[0],
        depth_size=sizes[-1][0],
        batch_size=batch_size,
        alpha=0.1,
        beta=0.1,
        use_peepholes=args.use_peepholes,
        loss_scales={
            'ref': args.loss_ref,
            'vrms': args.loss_vrms,
            'vint': args.loss_vint,
            'vdepth': args.loss_vdepth,
        },
        out_names={'ref', 'vrms', 'vint', 'vdepth'},
    )

    # Train the model.
    if args.training in [1, 2]:
        trainer = Trainer(
            nn=nn,
            case=case,
            checkpoint_dir=args.logdir,
            learning_rate=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.eps,
        )
        trainer.train_model(niter=args.epochs)

    # Test model.
    if args.training == 3:
        tester = Tester(nn=nn, case=case)
        savepath = os.path.join(case.datatest, "pred")
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        restore_from = tf.train.latest_checkpoint(args.logdir)
        tester.test_dataset(
            savepath=savepath,
            toeval={
                'ref': nn.outputs['ref'],
                'vrms': nn.outputs['vrms'],
                'vint': nn.outputs['vint'],
            },
            restore_from=restore_from,
        )

        if args.plot:
            tester.animated_predictions(
                labelnames=["ref", 'vrms', 'vint'],
                savepath=savepath,
                image=False,
            )
