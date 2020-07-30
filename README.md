[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3492115.svg)](https://doi.org/10.5281/zenodo.3492115)

# 2D Velocity estimation using neural-networks

Repository to build a neural network for 2D velocity model prediction from
seismic data.

## Contributing

This repository is organized in the following fashion. From highest level to
lowest:

*   The file [Case2Dtest.py](Case2Dtest.py) shows an example of how to generate
a training set, and train a NN with it. This is the starting point.
*   A neural network is defined in [RCNN2D.py](vrmslearn/RCNN2D.py).
This class builds the NN and the loss. It is used in [Case2Dtest.py](Case2Dtest.py).
To build a new network, a child class can be defined from RCNN2D.
*   To help with training, a class [Trainer](vrmslearn/Trainer.py) is provided.
It needs a Case class and a RCNN2D-like class as input.
*   The file [Case_define.py](Case_define.py) is where different cases are
defined.
*  A Case is implemented with the [Case](vrmslearn/Case.py) class. It provides an
interface to generate 2D velocity models and model the seismic data with
fixed parameters.
*  The Case class contain a [ModelParameters](vrmslearn/ModelParameters.py)
which regroup all parameters for model creation and data creation. A new Case
is usually built by changing those parameters.
*   The Case class contains the method 'generate_dataset' defined in
[DatasetGenerator.py](vrmslearn/DatasetGenerator.py), which allows computing
the seismic data on multiple GPUs. This creates the training, testing and validation
sets of a Case.
*   [DatasetGenerator.py](vrmslearn/DatasetGenerator.py) builds on two classes:
[ModelGenerator](vrmslearn/ModelGenerator.py) and the
[SeismicGenerator](vrmslearn/SeismicGenerator.py) that respectively generate
a velocity model and the seismic data.


## Installation

IMPORTANT: USE THE LATEST SEISCL VERSION ON THE DEVEL BRANCH!


You should clone this repository

    git clone https://github.com/gfabieno/SeisCL.git

#### a) Use Docker (easiest)

We provide a Docker image that contains all necessary python libraries like Tensorflow
and the seismic modeling code SeisCL.

You first need to install the Docker Engine, following the instructions [here](https://docs.docker.com/install/).
To use GPUs, you also need to install the [Nvidia docker](https://github.com/NVIDIA/nvidia-docker).
For the later to work, Nvidia drivers should be installed.
Then, when in the project repository, build the docker image as follows:

    docker build -t seisai:v0

You can then launch any of the python scripts in this repo as follows:

    docker run --gpus all -it\
               -v `pwd`:`pwd` -w `pwd` \
               --user $(id -u):$(id -g) \
               seisai:v0 Case_article.py --logdir=./Case_article

This makes accessible all gpus (`--gpus all`), mounting the current directory to a
to the same path in the docker (second line), running the docker as the current user
(for file permission), and runs the script `Case_article.py`.

#### b) Install all requirements

It is recommended to create a new virtual environment for this project with Python3.
The main python requirements are:
*   [tensorflow](https://www.tensorflow.org). This project was tested with versions 1.8 to 1.15.
The preferred method of installation is through pip, but many options are available.
*  [SeisCL](https://github.com/gfabieno/SeisCL). Follow the instruction in the README of
the SeisCL repository. Preferred compiling options for this project are api=opencl (use
OpenCL, which is faster than CUDA for small models) and nompi=1, because no MPI parallelization is required.
Be sure to install SeisCL's python wrapper.

Once SeisCL is installed, you can install all other python requirements with

    pip install -r requirements.txt
