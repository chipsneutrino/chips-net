# chips-net

[![Pipeline](https://gitlab.com/chipsneutrino/chips-net/badges/master/pipeline.svg)](https://gitlab.com/chipsneutrino/chips-net/pipelines)        [![Docs chipsneutrino.gitlab.io/chips-net](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://chipsneutrino.gitlab.io/chips-net/)

chips-net (CHIPS Convolutional Visual Network) provides the code to train Convolutional Neural Networks (CNNs) for CHIPS event analysis using the [Tensorflow](https://www.tensorflow.org/) framework.

The docs can be found at [https://chipsneutrino.gitlab.io/chips-net/](https://chipsneutrino.gitlab.io/chips-net/)

## Setup
To setup the correct version of CUDA and install the conda environment, run... 

```
$ source setup.sh
```

Once this is done you can source 'setup.sh' again in the future to activate the environment.

This also sets up two aliases (run and preprocess) used to execute run.py and preprocess.py.

## Preprocessing
To preprocess .root map files into tfrecords run...

```
$ preprocess [input_prod_directory]
```

## To Run
To run train, study, or evaluate, modify or create a new config file, exampes of which are in ./data/config/ and call...

```
$ run [yaml_configuration]
```

## Tips
You can use ./scripts/preprocess_maps.sh to preprocess a whole production set

It's important to first copy data to the local GPU machine or the training rate is reduced significantly due to the network bottleneck. The script ./scripts/copy_to_local.sh shows how to do this.

## Notebooks
To use the notebooks in ./scripts/ run the following and navigate to the url given.

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