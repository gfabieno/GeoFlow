# -*- coding: utf-8 -*-

from GeoFlow.__main__ import parser
from GeoFlow.AutomatedTraining import optimize
from GeoFlow import DefinedDataset, DefinedNN


args, config = parser.parse_known_args()
config = {name[2:]: eval(value)
          for name, value in zip(config[::2], config[1::2])}
args.nn = getattr(DefinedNN, args.nn)
args.params = getattr(DefinedNN, args.params)(is_training=args.train)
args.dataset = getattr(DefinedDataset, args.dataset)(args.noise)

if args.debug:
    args.params["epochs"] = 1
    args.params["steps_per_epoch"] = 5

optimize(args, **config)
