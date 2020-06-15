# chips-net

[![Pipeline](https://gitlab.com/chipsneutrino/chips-net/badges/master/pipeline.svg)](https://gitlab.com/chipsneutrino/chips-net/pipelines) [![Docs chipsneutrino.gitlab.io/chips-net](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://chipsneutrino.gitlab.io/chips-net/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

The CHIPS convolutional visual network provides a framework to train and evaluate CNNs for neutrino water Cherenkov event classification and reconstruction. The [Tensorflow](https://www.tensorflow.org/) framework is used as the backend.

The docs can be found at [https://chipsneutrino.gitlab.io/chips-net/](https://chipsneutrino.gitlab.io/chips-net/)

## Setup
To setup the correct version of CUDA and install the conda environment, run... 

```
$ source setup.sh
```

Once this is done you can source 'setup.sh' again in the future to activate the environment.

This also sets up the alias 'run' used to execute the run.py script that executes all possible jobs in chipsnet

## To Run
To run create data, train, study, or evaluate, modify or create a new config file, examples of which are in ./config/ and call...

```
$ run [config_path]
```

## Tips
It's important to first copy data to the local GPU machine or the training rate is reduced significantly due to the network bottleneck.

## Notebooks
To use the notebooks in ./notebooks/ run the following and navigate to the url given.

You will need to forward the port to your local machine if working remotely.

```
$ jupyter-notebook
```

## Tensorboard
To run tensorboard on the outputs from a model use...

```
tensorboard --logdir [model_dir]/tensorboard
```

Again you will need to forward the port to your local machine if working remotely.

## Jupyter Slideshow
To serve a jupyter notebook as a slideshow run...

Again you will need to forward the port to your local machine if working remotely.

```
$ jupyter nbconvert [notebook] --to slides --post serve
```

## Comet Integration

To enable this feature define a file .comet containing 

    COMET_API_KEY=<your-comet-api-key>
    COMET_WORKSPACE=<your-comet-workspace>
    COMET_PROJECT_NAME=<your-comet-project-name>