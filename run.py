"""Main training script

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This script is the main chips-cvn training script. Given the input
configuration it trains the given model and then evaluates the test
dataset. It can also carry out hyperparameter optimisation using
SHERPA which requires a modified configuration file.
"""

import argparse
import os
import logging

from comet_ml import Experiment
import tensorflow as tf

import chipscvn.config
import chipscvn.data
import chipscvn.models
import chipscvn.trainers
import chipscvn.studies
import chipscvn.utils


def train_model(config):
    """Trains a model according to the configuration."""
    try:
        Experiment()
    except Exception:
        print('Error: Need to set comet_ml env variables')
        pass

    chipscvn.config.setup_dirs(config, True)
    data = chipscvn.data.DataLoader(config)
    model = chipscvn.utils.get_model(config)
    model.summarise()
    trainer = chipscvn.utils.get_trainer(config, model, data)
    trainer.train()
    trainer.save()


def study_model(config):
    """Conducts a SHERPA study on a model according to the configuration."""
    chipscvn.config.setup_dirs(config, True)
    study = chipscvn.utils.get_study(config)
    study.run()


def evaluate_model(config):
    """Evaluate the trained model according to the configuration."""
    evaluator = chipscvn.utils.get_evaluator(config)
    evaluator.run()


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description='CHIPS CVN')
    parser.add_argument('config', help='path to the configuration file')
    parser.add_argument('--verbose', action='store_true', help='Make tensorflow verbose')
    return parser.parse_args()


def main():
    """Main function called by training script."""
    args = parse_args()
    if not args.verbose:  # Turn off Tensorflow logging by default
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.disable(logging.CRITICAL)

    print('\n--- Its Magic, it must be the CHIPS CVN ---')
    chipscvn.utils.gpu_setup()  # Setup the GPU's so they work on all machines
    config = chipscvn.config.get(args.config)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if config.task == 'train':
            train_model(config)
        elif config.task == 'study':
            study_model(config)
        elif config.task == 'evaluate':
            evaluate_model(config)
        else:
            print('\nError: must define a task in configuration [train, study, evaluate]')
            raise SystemExit


if __name__ == '__main__':
    main()
