# chips-net

[![Pipeline](https://gitlab.com/chipsneutrino/chips-net/badges/master/pipeline.svg)](https://gitlab.com/chipsneutrino/chips-net/pipelines) [![Docs chipsneutrino.gitlab.io/chips-net](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://chipsneutrino.gitlab.io/chips-net/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/78709e60f88f4918be95f8dcbabe4dd0)](https://www.codacy.com/gl/chipsneutrino/chips-net?utm_source=gitlab.com&amp;utm_medium=referral&amp;utm_content=chipsneutrino/chips-net&amp;utm_campaign=Badge_Grade)

The CHIPS convolutional visual network provides a framework to train and evaluate CNNs for neutrino water Cherenkov event classification and reconstruction. The [Tensorflow](https://www.tensorflow.org/) framework is used as the backend.

The docs can be found at [https://chipsneutrino.gitlab.io/chips-net/](https://chipsneutrino.gitlab.io/chips-net/)

## Setup
To setup the correct version of CUDA and install the conda environment, run... 

```bash
source setup.sh
```

Once this is done you can source 'setup.sh' again in the future to activate the environment.

This also sets up the alias 'run' used to execute the run.py script that executes all possible jobs in chipsnet

## Usage
To run the example create, train and study configurations in ./config/ call...

```bash
make create
```

```bash
make train
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

You can run the full analysis notebook using [papermill](https://papermill.readthedocs.io/en/latest/index.html) with the following command...

```bash
papermill notebooks/analysis.ipynb notebooks/analysis_complete.ipynb \
    -p config_path "/mnt/storage/jtingey/chips-net/config/eval.yaml" \
    -p save_path "/mnt/storage/jtingey/chips-net/data/output/"
```

## Tips
It's important to first copy data to the local GPU machine or the training rate is reduced significantly due to the network bottleneck.

## Notebooks
To use the notebooks in ./notebooks/ run the following and navigate to the url given.

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

## Comet Integration

To enable this feature define a file .comet containing 

```text
COMET_API_KEY=<your-comet-api-key>
COMET_WORKSPACE=<your-comet-workspace>
COMET_PROJECT_NAME=<your-comet-project-name>
```