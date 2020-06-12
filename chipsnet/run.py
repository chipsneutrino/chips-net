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

import chipsnet.config  # noqa: E402
import chipsnet.data  # noqa: E402
import chipsnet.models  # noqa: E402
import chipsnet.trainer  # noqa: E402
import chipsnet.study  # noqa: E402


def setup_gpus():
    # Need to setup the GPU's before we import anything else that uses tensorflow
    gpus = tf.config.list_physical_devices('GPU')
    if tf.config.list_physical_devices('GPU'):
        try:  # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:  # Memory growth must be set before GPUs have been initialized
            print(e)


def create_data(config):
    """Preprocesses input .root files into .tfrecords ready for use in training.
    Args:
        config (dotmap.DotMap): Configuration namespace
    """
    print('--- Setting up data creator ---\n')
    creator = chipsnet.data.Creator(config)
    print('--- Running creation ---\n')
    creator.run()


def train_model(config):
    """Trains a model according to the configuration.
    Args:
        config (dotmap.DotMap): Configuration namespace
    """
    comet_exp = None
    if config.exp.comet:
        try:
            comet_exp = Experiment()
        except Exception:
            print('Error: Need to set comet_ml env variables')
            pass

    setup_gpus()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        print('--- Setting up directories ---\n')
        chipsnet.config.setup_dirs(config, True)
        print('--- Setting up data reader ---\n')
        data = chipsnet.data.Reader(config)
        print('--- Building model ---\n')
        model = chipsnet.models.Model(config)
        if config.trainer.epochs > 0:
            print('\n--- Training model ---')
            trainer = chipsnet.trainer.Trainer(config, model, data)
            trainer.train()
            print('\n--- Running quick evaluation ---\n')
            trainer.eval()
            print('\n--- Saving model to {} ---\n'.format(config.exp.exp_dir))
            trainer.save()
        else:
            print('\n--- Skipping training ---\n')

    if config.exp.comet:
        try:
            comet_exp.end()
        except Exception:
            pass


def study_model(config):
    """Conducts a SHERPA study on a model according to the configuration.
    Args:
        config (dotmap.DotMap): Configuration namespace
    """
    comet_exp = None
    if config.exp.comet:
        try:
            comet_exp = Experiment()
        except Exception:
            print('Error: Need to set comet_ml env variables')
            pass

    setup_gpus()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        print('--- Setting up directories ---\n')
        chipsnet.config.setup_dirs(config, True)
        print('--- Setting up the study---\n')
        study = chipsnet.study.SherpaStudy(config)
        print('--- Running study ---\n')
        study.run()

    if config.exp.comet:
        try:
            comet_exp.end()
        except Exception:
            pass


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

    if config.task == 'create':
        create_data(config)
    elif config.task == 'train':
        train_model(config)
    elif config.task == 'study':
        study_model(config)
    else:
        print('\nError: must define a task in configuration [create, train, study]')
        raise SystemExit
    print('--- Magic complete ---\n')


if __name__ == '__main__':
    main()
