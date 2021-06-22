# chips-net

[![Pipeline](https://gitlab.com/chipsneutrino/chips-net/badges/master/pipeline.svg)](https://gitlab.com/chipsneutrino/chips-net/pipelines) 
[![Docs chipsneutrino.gitlab.io/chips-net](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://chipsneutrino.gitlab.io/chips-net/) 
[![Code style:
black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the software to train and evaluate Convolutional Neural
Networks (CNNs) for neutrino event characterisation (cosmic muon rejection, beam
neutrino classification, neutrino energy estimation) within water Cherenkov
detectors, specifically for CHIPS R&D project detectors. For a detailed
description of the implementation see
[link](https://joshtingey.github.io/phd-thesis/thesis.pdf). We use the
[Tensorflow](https://www.tensorflow.org/) framework as the backend and employ
multiple GPUs for training and inference.

## Setup

The "environment.yaml" file contains the required Conda environment definition. Additionally, CUDA version 10.1 is required. To ensure that this version is used and the environment is installed correctly, we provide a bash script that can be used as follows:

```bash
source setup.sh
```

Once this is complete, you can source the same script again in the future to activate the environment.

## Usage

The default configurations for creating data, running training, and performing SHERPA hyperparameter optimisation studies are located in ./chipsnet/config/ these will need to be modified for your setup.


To create the training data (.tfrecords) files from the simulation 'map' files run:

```bash
make create
```

To train a beam neutrino classification model run:

```bash
make train_beam
```

To train a beam neutrino energy estimation regression model run:

```bash
make train_energy
```

To run a SHERPA hyperparameter optimisation study run:

```bash
make study
```

To run any task with a given configuration use the following:

```bash
python chipsnet/run.py [config_path]
```

To run all the tests call:

```bash
make test
```

## Analysis

To use the analysis results notebook in ./analysis/ run the following and navigate to the url given.

You will need to forward the port to your local machine if working remotely.

```bash
jupyter-lab
```

## Tips

- It's important to first copy data to the local GPU machine or the training rate is reduced significantly due to the networking bottleneck.
