from GeoFlow.__main__ import parse_args
from .AutomatedTraining import optimize

args = parse_args()

if args.debug:
    args.params["epochs"] = 1
    args.params["steps_per_epoch"] = 5

optimize(args, **args.params)
