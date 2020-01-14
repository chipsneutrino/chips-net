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
$ python data.py -i [input_directory] -o [output_directory]
```

## Training
To train a model run...

```
$ python train.py -i [input_directory] -o [output_directory] -c [json_configuration]
```