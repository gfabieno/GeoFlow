#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Script to perform data creation and training for the main case presented in
    the article. For a smaller training set, see Case_small.py
"""

from vrmslearn.ModelParameters import ModelParameters
from .Cases_define import Case_1Darticle
from vrmslearn.SeismicGenerator import SeismicGenerator, generate_dataset
from vrmslearn.Trainer import Trainer
from vrmslearn.RCNN import RCNN
import os
import argparse
import tensorflow as tf
import fnmatch

if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to parse for training
    parser.add_argument(
        "--nthread",
        type=int,
        default=1,
        help="Number of threads for data creation"
    )
    parser.add_argument(
        "--nthread_read",
        type=int,
        default=1,
        help="Number of threads used as input producer"
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
        default=1,
        help="1: training only, 0: create dataset only, 2: training+dataset"
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="./seiscl_workdir",
        help="name of SeisCL working directory "
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
        "--noise",
        type=int,
        default=1,
        help="1: Add noise to the data"
    )
    parser.add_argument(
        "--use_peepholes",
        type=int,
        default=1,
        help="1: Use peephole version of LSTM"
    )

    # Parse the input for training parameters
    args, unparsed = parser.parse_known_args()

    savepath = "./dataset_article"
    logdir = args.logdir
    nthread = args.nthread
    batch_size = args.batchsize

    """
        _______________________Define the parameters ______________________
    """
    pars = Case_1Darticle(noise=args.noise)

    """
        _______________________Generate the dataset_____________________________
    """
    gen = SeismicGenerator(model_parameters=pars)

    pars.num_layers = 0
    dhmins = [5]
    layer_num_mins = [5, 10, 30, 50]
    nexamples = 10000

    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    if args.training != 1:
        for dhmin in dhmins:
            for layer_num_min in layer_num_mins:
                pars.layer_dh_min = dhmin
                pars.layer_num_min = layer_num_min
                this_savepath = (savepath
                                 + "/dhmin%d" % dhmin
                                 + "_layer_num_min%d" % layer_num_min)
                generate_dataset(pars=pars,
                                 savepath=this_savepath,
                                 nthread=args.nthread,
                                 nexamples=nexamples,
                                 workdir=args.workdir)