# -*- coding: utf-8 -*-

"""Main running script

This script is the main chips-cvn training script. Given the input
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

import chipscvn.config  # noqa: E402
import chipscvn.data  # noqa: E402
import chipscvn.models  # noqa: E402
import chipscvn.trainers  # noqa: E402
import chipscvn.studies  # noqa: E402
import chipscvn.evaluator  # noqa: E402


def train_model(config, strategy):
    """Trains a model according to the configuration.
    Args:
        config (dotmap.DotMap): Configuration namespace
    """
    try:
        Experiment()
    except Exception:
        print('Error: Need to set comet_ml env variables')
        pass

    chipscvn.config.setup_dirs(config, True)
    data = chipscvn.data.DataLoader(config)
    model = chipscvn.models.get_model(config)
    trainer = chipscvn.trainers.get_trainer(config, model, data, strategy)
    trainer.train()
    trainer.save()


def study_model(config, strategy):
    """Conducts a SHERPA study on a model according to the configuration.
    Args:
        config (dotmap.DotMap): Configuration namespace
    """
    chipscvn.config.setup_dirs(config, True)
    study = chipscvn.studies.get_study(config)
    study.run()


def evaluate_model(config, strategy):
    """Evaluate the trained model according to the configuration.
    Args:
        config (dotmap.DotMap): Configuration namespace
    """
    evaluator = chipscvn.evaluator.Evaluator(config)
    evaluator.run_all()


def parse_args():
    """Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description='CHIPS CVN')
    parser.add_argument('config', help='path to the configuration file')
    return parser.parse_args()


def main():
    """Main function called by the run script.
    """
    print('\n--- Its Magic, it must be the CHIPS CVN ---')
    config = chipscvn.config.get(parse_args().config)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if config.task == 'train':
            train_model(config, strategy)
        elif config.task == 'study':
            study_model(config, strategy)
        elif config.task == 'evaluate':
            evaluate_model(config, strategy)
        else:
            print('\nError: must define a task in configuration [train, study, evaluate]')
            raise SystemExit


if __name__ == '__main__':
    main()
