[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3492115.svg)](https://doi.org/10.5281/zenodo.3492115)

# GeoFlow

Dataset management interface with Keras for geophysics

## Installation

**IMPORTANT: Use the latest SeisCL version on the `devel` branch.**


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
               seisai:v0 main.py

This makes accessible all gpus (`--gpus all`), mounting the current directory to a
to the same path in the docker (second line), running the docker as the current user
(for file permission), and runs the script `Case_article.py`.

#### b) Install all requirements

It is recommended to create a new virtual environment for this project with Python3.
Install all requirements with
```
pip install .
```
The main Python requirements are:
* [tensorflow](https://www.tensorflow.org)
* [SeisCL](https://github.com/gfabieno/SeisCL). Navigate to the directory of the SeisCL installation and follow the instruction in the README. Preferred compiling options for this project are api=opencl (use OpenCL, which is faster than CUDA for small models) and nompi=1, because no MPI parallelization is required. Be sure to install SeisCL's python wrapper.


## Getting started

### Style guide

Code style should follow closely PEP 8. Commits should follow closely `git` good practices. 
Refer to [`STYLEGUIDE.md`](https://github.com/gfabieno/Deep_2D_velocity/blob/master/STYLEGUIDE.md) for a comprehensive guide to this project's style conventions.

### Package structure

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

### Creating a Geodataset

We show how to create a Geodaset with Geoflow. You can look at other scripts in 
the `DefinedDataset` directory for other examples. 

#### Step 1 - Create your dataset script

Start by creating a new script for your project in `DefinedDataset`. In this script, 
you can create new derived classes from the classes [EarthModel](GeoFlow/EarthModel.py), 
[Aquisition](GeoFlow/SeismicGenerator.py) and [GeoDataset](GeoFlow/GeoDataset.py). 
To do so, you need to import the base classes.

```python
from GeoFlow import EarthModel
from GeoFlow import Acquisition
from GeoFlow import GeoDataset
```

#### Step 2 - Override `build_stratigraphy` from [EarthModel](GeoFlow/EarthModel.py)
You'll probably need to define your own stratigraphy for your synthetic dataset. To do so, 
you need to override the `build_stratigraphy` method in <u>your</u> new class from your script.

```python
class NewModel(EarthModel):
    def build_stratigraphy(self):
        ...
```

#### Step 3 - Override `set_rec_src` from [Aquisition](GeoFlow/SeismicGenerator.py)
You can also change the number of receivers, their spacing and the source position. To do so, 
you need to override the `set_rec_src` method in <u>your</u> new class from your script.
 
```python  
class NewAquisition(Aquisition):
    def set_rec_src(self):
        ...
```
#### Step 4 - Define the inputs and outputs  

#### Step 5 - Override `set_dataset` from [GeoDataset](GeoFlow/GeoDataset.py)
Finally you need to override the `set_dataset` method in <u>your</u> new class from your script.
This method allows you to choose the size of the synthetic dataset and to choose all of your 
parameters. In this method you will also define the inputs and outputs to save in the 
hdf5 example files.


```python    
class DatasetExample(GeoDataset):
    name = "DatasetExample"
    
    def set_dataset(self):
        self.trainsize = 10000
        self.validatesize = 100
        self.testsize = 100
        
        model = NewModel()
        ...  
        acquire = NewAquisition(model=model)
        ...
        inputs=...
        outputs=...
```

#### Step 6 - Run [main](main.py) script to create dataset
The [main](main.py) script allows to generate and view any dataset found in 
[DefinedDataset](DefinedDataset/). To select a dataset, use the option `dataset`
with the name of your dataset. 

    python main.py --dataset=DatasetExample --plot --debug

The `--debug` option create a dataset of only 5 examples and the `--plot` option
create an animation of the dataset. The [main](main.py) script can be used also
for training a Neural Network on a specific dataset, as shown in the next
section.

### Defining and training a Neural Net 


    
