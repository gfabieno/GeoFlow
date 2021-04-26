from GeoFlow.__main__ import parse_args
from .AutomatedTraining import optimize

args = parse_args()

if args.debug:
    args.params["epochs"] = 1
    args.params["steps_per_epoch"] = 5

optimize(nn=args.nn,
         params=args.params,
         dataset=args.dataset,
         logdir=args.logdir,
         gpus=args.gpus,
         debug=args.debug,
         eager=args.eager,
         **args.params)
