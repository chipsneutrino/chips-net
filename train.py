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

import tensorflow as tf

import chipscvn.config
import chipscvn.data
import chipscvn.models
import chipscvn.trainers
import chipscvn.studies
import chipscvn.utils


def train_model(config):
    """Trains a model according to the configuration."""
    data = chipscvn.data.DataLoader(config)
    model = chipscvn.utils.get_model(config)
    model.summarise()
    trainer = chipscvn.utils.get_trainer(config, model, data)
    trainer.train()


def study_model(config):
    """Conducts a SHERPA study on a model according to the configuration."""
    study = chipscvn.utils.get_study(config)
    study.run()


def test_model(config):
    """Evaluate the trained model according to the configuration."""


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description='CHIPS CVN')
    parser.add_argument('config', help='path to the configuration file')
    parser.add_argument('--train', action='store_true', help='train the configuration model')
    parser.add_argument('--study', action='store_true', help='study the configuration model')
    parser.add_argument('--evaluate', action='store_true', help='test the configuration model')
    parser.add_argument('--verbose', action='store_true', help='Make tensorflow verbose')
    return parser.parse_args()


def main():
    """Main function called by training script."""
    args = parse_args()
    if not args.verbose:  # Turn off Tensorflow logging by default
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.disable(logging.CRITICAL)

    print("\n--- It's Magic, it must be the CHIPS CVN ---")
    chipscvn.utils.gpu_setup()  # Setup the GPU's so they work on all machines
    config = chipscvn.config.process_config(args.config)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if args.train:
            train_model(config)
        elif args.study:
            study_model(config)
        elif args.test:
            test_model(config)
        else:
            print("\nError: must select task [train, study, test]")
            raise SystemExit


if __name__ == '__main__':
    main()
