# -*- coding: utf-8 -*-

"""Main running script

This script is the main chipsnet training script. Given the input
configuration it trains the given model and then evaluates the test
dataset. It can also carry out hyperparameter optimisation using
SHERPA which requires a modified configuration file.

Example:
    The example below runs a cosmic model training
    $ python run.py ./config/cosmic.yml
"""

import argparse
import os
import logging

from comet_ml import Experiment
import tensorflow as tf

# Need to setup the logging level before we use tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.CRITICAL)

# Need to setup the GPU's before we import anything else that uses tensorflow
gpus = tf.config.list_physical_devices('GPU')
if tf.config.list_physical_devices('GPU'):
    try:  # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
    except RuntimeError as e:  # Memory growth must be set before GPUs have been initialized
        print(e)

import chipsnet.config  # noqa: E402
import chipsnet.data  # noqa: E402
import chipsnet.models  # noqa: E402
import chipsnet.trainers  # noqa: E402
import chipsnet.studies  # noqa: E402
import chipsnet.evaluator  # noqa: E402


def create_data(config):
    """Preprocesses input .root files into .tfrecords ready for use in training.
    Args:
        config (dotmap.DotMap): Configuration namespace
    """
    creator = chipsnet.data.Creator(config)
    creator.run()


def train_model(config):
    """Trains a model according to the configuration.
    Args:
        config (dotmap.DotMap): Configuration namespace
    """
    try:
        experiment = Experiment()
    except Exception:
        print('Error: Need to set comet_ml env variables')
        pass

    print('--- Setting up directories ---\n')
    chipsnet.config.setup_dirs(config, True)
    print('--- Setting up data loader ---\n')
    data = chipsnet.data.Loader(config)
    print('--- Building model ---\n')
    model = chipsnet.models.get_model(config)
    if config.trainer.epochs > 0:
        print('\n--- Training model ---')
        trainer = chipsnet.trainers.get_trainer(config, model, data)
        trainer.train()
        print('\n--- Running quick evaluation ---\n')
        trainer.eval()
        print('\n--- Saving model ---\n')
        trainer.save()
    else:
        print('\n--- Skipping training ---\n')

    try:
        experiment.end()
    except Exception:
        pass


def study_model(config):
    """Conducts a SHERPA study on a model according to the configuration.
    Args:
        config (dotmap.DotMap): Configuration namespace
    """
    print('--- Setting up directories ---\n')
    chipsnet.config.setup_dirs(config, True)
    print('--- Setting up data loader ---\n')
    study = chipsnet.studies.get_study(config)
    print('--- Running study ---\n')
    study.run()


def evaluate_model(config):
    """Evaluate the trained model according to the configuration.
    Args:
        config (dotmap.DotMap): Configuration namespace
    """
    print('--- Setting up evaluator ---\n')
    evaluator = chipsnet.evaluator.Evaluator(config)
    evaluator.run_all()


def parse_args():
    """Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description='chipsnet')
    parser.add_argument('config', help='path to the configuration file')
    return parser.parse_args()


def main():
    """Main function called by the run script.
    """
    print('\n--- Its Magic, it must be chipsnet ---\n')
    config = chipsnet.config.get(parse_args().config)

    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    if config.task == 'create':
        create_data(config)
    elif config.task == 'train':
        train_model(config)
    elif config.task == 'study':
        study_model(config)
    elif config.task == 'evaluate':
        evaluate_model(config)
    else:
        print('\nError: must define a task in configuration [create, train, study, evaluate]')
        raise SystemExit
    print('--- Magic complete ---\n')


if __name__ == '__main__':
    main()