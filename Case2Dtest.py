from Cases_define import Case_2Dtest, Case_1Dsmall
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

    # Parse the input for training parameters
    args, unparsed = parser.parse_known_args()

    logdir = args.logdir
    batch_size = args.batchsize

    """
        _______________________Define the parameters ______________________
    """
    #TODO Input argument choosing which case to run

    # case = Case_2Dtest(
    #     noise=args.noise,
    #     trainsize=10000,
    #     validatesize=1000,
    #     testsize=1000,
    # )
    case = Case_1Dsmall(
        trainsize=100,
        validatesize=10,
        testsize=10)
    if args.training == 3 and case.testsize < batch_size:
        batch_size = case.testsize

    """
        _______________________Generate the dataset________________________
    """

    if args.training in [0, 2]:
        case.generate_dataset(ngpu=args.ngpu)

    if args.plot:
        case.animated_dataset()

    # TODO Test different loss_scales
    sizes = case.get_dimensions()
    nn = RCNN2D(input_size=sizes[0],
                depth_size=sizes[-1][0],
                batch_size=batch_size,
                alpha=0.1,
                beta=0.1,
                use_peepholes=args.use_peepholes,
                loss_scales={
                    'ref': args.loss_ref,
                    'vrms': args.loss_vrms,
                    'vint': args.loss_vint,
                },
                out_names={'ref', 'vrms', 'vint'})
    """
        _______________________Train the model_____________________________
    """
    if args.training in [1, 2]:
        trainer = Trainer(nn=nn,
                          case=case,
                          checkpoint_dir=args.logdir,
                          learning_rate=args.lr,
                          beta1=args.beta1,
                          beta2=args.beta2,
                          epsilon=args.eps)

        trainer.train_model(niter=args.epochs)
    """
        _______________________Validate results_____________________________
    """
    if args.training == 3:
        tester = Tester(nn=nn, case=case)
        savepath = os.path.join(case.datatest, "pred")
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        restore_from = tf.train.latest_checkpoint(args.logdir)
        tester.test_dataset(savepath=savepath,
                            toeval={'ref': nn.outputs['ref'],
                                    'vrms': nn.outputs['vrms'],
                                    'vint': nn.outputs['vint']},
                            restore_from=restore_from)

        if args.plot:
            tester.animated_predictions(labelnames=["ref", 'vrms', 'vint'],
                                        savepath=savepath,
                                        image=False)
