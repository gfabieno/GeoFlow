[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3492115.svg)](https://doi.org/10.5281/zenodo.3492115)

# 2D Velocity estimation using neural-networks

Repository to build a neural network for 2D velocity model prediction from
seismic data.

## Contributing

This repository is organized in the following fashion. From highest level to
lowest:

*   The file [Dataset2Dtest.py](main.py) shows an example of how to generate
a training set, and train a NN with it. This is the starting point.
*   A neural network is defined in [RCNN2D.py](GeoFlow/RCNN2D.py).
This class builds the NN and the loss. It is used in [Dataset2Dtest.py](main.py).
To build a new network, a child class can be defined from `RCNN2D`.
*   To help with training, a class [Trainer](GeoFlow/Trainer.py) is provided.
It needs a GeoDataset class and a `RCNN2D`-like class as input.
*   The file [Dataset_define.py](Dataset_define.py) is where different Datasets are
defined.
*  A `GeoDataset` is implemented with the [GeoDataset](GeoFlow/GeoDataset.py) class. It provides an
interface to generate 2D velocity models and model the seismic data with
fixed parameters.
*   The `GeoDataset` class contains the method `set_dataset`. A new GeoDataset can be defined 
by defining a child class from the `GeoDataset` base class and overriding `set_dataset`.
This method needs to return three objects based on three classes
    *  [EarthModel](GeoFlow/BaseModelGenerator.py). This class allows
    to generate a random model. It is based the ModGen library available upon
    request. Different model generator can be defined from this case (see
    [MarineModel](GeoFlow/BaseModelGenerator.py)).
    *   [Acquisition](GeoFlow/SeismicGenerator.py) defines all the parameters
    for the creation of the seismic data by SeisCL. In particular, override the
    method `set_rec_src` to define a different acquisition setup.
    * [OutputGenerator](GeoFlow/GraphIO.py) is a class that generate the
    labels from the model and acquires objects. 


#### Style guide

Code style should follow PEP 8. Hanging indents should be at the same level as
the opening parenthesis, bracket or brace, as in:
```
parser.add_argument("--case",
                    type=str,
                    default="Case1Dsmall",
                    help="Name of the case from `Cases_define` to use")
```
Identifiers must be descriptive and short enough to maintain ease-of-use.

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
