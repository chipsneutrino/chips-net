# chips-net

[![Pipeline](https://gitlab.com/chipsneutrino/chips-net/badges/master/pipeline.svg)](https://gitlab.com/chipsneutrino/chips-net/pipelines) 
[![Docs chipsneutrino.gitlab.io/chips-net](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://chipsneutrino.gitlab.io/chips-net/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

The CHIPS convolutional visual network provides a framework to train and evaluate CNNs for neutrino water Cherenkov event classification and reconstruction. The [Tensorflow](https://www.tensorflow.org/) framework is used as the backend.

The docs can be found at [https://chipsneutrino.gitlab.io/chips-net/](https://chipsneutrino.gitlab.io/chips-net/)

## Setup
To setup the correct version of CUDA and install the conda environment, run... 

```bash
source setup.sh
```

Once this is done you can source 'setup.sh' again in the future to activate the environment.

## Usage
To run the example create, train and study configurations in ./chipsnet/config/ call...

```bash
make create
```

```bash
make train_beam
```

```bash
make study
```

To run a new configuration task use the following...

```bash
python chipsnet/run.py [config_path]
```

To run all the tests call...

```bash
make test
```

## Tips
It's important to first copy data to the local GPU machine or the training rate is reduced significantly due to the network bottleneck.

## Analysis
To use the analysis results notebook in ./analysis/ run the following and navigate to the url given.

You will need to forward the port to your local machine if working remotely.

```bash
jupyter-lab
```

## Tensorboard
To run tensorboard on the outputs from a model use...

```bash
tensorboard --logdir [model_dir]/tensorboard
```

Again you will need to forward the port to your local machine if working remotely.

## Jupyter Slideshow
To serve a jupyter notebook as a slideshow run...

Again you will need to forward the port to your local machine if working remotely.

```bash
jupyter nbconvert [notebook] --to slides --post serve
```