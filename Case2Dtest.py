from Cases_define import Case_2Dtest
from vrmslearn.RCNN2D import RCNN2D
from vrmslearn.Trainer import Trainer
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

    # Parse the input for training parameters
    args, unparsed = parser.parse_known_args()

    logdir = args.logdir
    batch_size = args.batchsize

    """
        _______________________Define the parameters ______________________
    """
    case = Case_2Dtest(noise=args.noise)

    """
        _______________________Generate the dataset________________________
    """

    if args.training != 1:
        case.generate_dataset(ngpu=args.ngpu)

    case.plot_example()
    case.plot_model()
    """
        _______________________Train the model_____________________________
    """
    input_size, label_size = case.get_dimensions()
    nn = RCNN2D(input_size=input_size,
                label_size=label_size,
                batch_size=batch_size,
                alpha=0.1,
                beta=0.1,
                use_peepholes=args.use_peepholes)

    trainer = Trainer(nn=nn,
                      case=case,
                      checkpoint_dir=args.logdir,
                      learning_rate=args.lr,
                      beta1=args.beta1,
                      beta2=args.beta2,
                      epsilon=args.eps)

    trainer.train_model(niter=args.epochs)
