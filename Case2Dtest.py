from Cases_define import Case_2Dtest
import argparse


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
        help="1: training only, 0: create dataset only, 2: training+dataset"
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
        default=40,
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
        default=1,
        help="1: Add noise to the data"
        )


    # Parse the input for training parameters
    args, unparsed = parser.parse_known_args()

    savepath = "./Case2Dtest"
    logdir = args.logdir
    batch_size = args.batchsize

    """
        _______________________Define the parameters ______________________
    """
    case = Case_2Dtest(noise=args.noise, trainsize=1)

    """
        _______________________Generate the dataset_____________________________
    """

    if args.training != 1:
        case.generate_dataset(ngpu=args.ngpu)



