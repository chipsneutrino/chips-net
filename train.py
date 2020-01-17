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
import sherpa
import config
import data
import models
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def run_study(config):
    """Creates a SHERPA hyperparameter study."""
    pars = models.SingleParModel.study_parameters(config)
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=config.trials)
    study = sherpa.Study(parameters=pars, algorithm=algorithm, lower_is_better=False,
                         output_dir=config.exp_dir)

    for trial in study:
        # Adjust the configuration for this trial
        for key in trial.parameters.keys():
            config[key] = trial.parameters[key]

        train_ds, val_ds, test_ds = data.datasets(config.input_dirs, config.img_shape)

        model = models.SingleParModel(config)
        cb = [study.keras_callback(trial, objective_name='val_mae')]
        model.fit(train_ds, val_ds, cb)
        study.finalize(trial)
    study.save()


def train_model(config)


def evaluate_model(model, test_ds):
    """Evaluate the trained model on the test dataset."""
    test_ds = test_ds.batch(64)
    predictions = model.model.predict(test_ds)
    print(predictions.shape)


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description='CHIPS CVN')
    parser.add_argument('config', help='path to the configuration file')
    parser.add_argument('--study', action='store_true', help='Are we running a sherpa study?')
    return parser.parse_args()


def main():
    """Main function called by script."""
    print("\nCHIPS CVN - It's Magic\n")

    args = parse_args()
    config = config.process_config(args.config)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if args.study:
            run_study(config)
        else:
            train_ds, val_ds, test_ds = data.datasets(config.input_dirs, config.img_shape)
            model = models.SingleParModel(config)
            train_ds.cache()
            val_ds.cache()
            model.fit(train_ds, val_ds)
            evaluate_model(model, test_ds)


if __name__ == '__main__':
    main()
