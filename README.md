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

## Training
To train a model run...

```
$ python run.py [yaml_configuration] --train
```

## SHERPA Study
To run a study...

```
$ python run.py [yaml_configuration] --study
```

## SHERPA Study
To use eval and explain notebooks...

```
$ jupyter-notebook
```

and navigate to the link, you will need to make sure you are forwarding the port

## Jupyter Slideshow
To serve the slideshow...

```
$ jupyter nbconvert [notebook] --to slides --post serve
```