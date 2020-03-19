# chips-cvn
chips-cvn (CHIPS Convolutional Visual Network) provides the code to train Convolutional Neural Networks (CNNs) for CHIPS event analysis using the tensorflow framework.

## Setup
To install the conda environment with all the required dependencies run...

```
$ source setup.sh
```

Once this is done you can run setup.sh again in the future to activate the environment.

## Preprocessing
To preprocess .root map files into tfrecords run...

```
$ python preprocess.py [input_directory] --reduce
```

## To Run
To run train, study, or evaluate, modify config file and call...

```
$ python run.py [yaml_configuration]
```

## Notebooks
To use evaluate and explain notebooks...

```
$ jupyter-notebook
```

and navigate to the link, you will need to make sure you are forwarding the port

## Jupyter Slideshow
To serve the slideshow...

```
$ jupyter nbconvert [notebook] --to slides --post serve
```