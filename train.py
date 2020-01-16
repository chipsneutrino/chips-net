"""Main training script

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This script is the main chips-cvn training script. Given the input
configuration it trains the given model and then evaluates the test 
dataset. It can also carry out hyperparameter optimisation using 
SHERPA which requires a modified configuration file.
"""

import argparse
import tensorflow as tf
import sherpa
import config
import data
import models


def make_study(conf):
    """Creates a SHERPA hyperparameter study."""
    pars = models.SingleParModel.study_parameters(conf)
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=conf.trials)
    study = sherpa.Study(parameters=pars, algorithm=algorithm,
                         lower_is_better=False, output_dir=conf.exp_dir)
    return study


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description='CHIPS CVN')
    parser.add_argument('config', help='path to the configuration file')
    parser.add_argument('--study', action='store_true',
                        help='Are we running a sherpa study?')
    return parser.parse_args()


def main():
    """Main function called by script."""
    print("\nCHIPS CVN - It's Magic\n")

    args = parse_args()
    conf = config.process_config(args.config)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if args.study:
            study = make_study(conf)
            for trial in study:
                # Adjust the configuration for this trial
                for key in trial.parameters.keys():
                    conf[key] = trial.parameters[key]

                train_ds, val_ds, test_ds = data.datasets(conf.input_dirs,
                                                          conf.img_shape)

                model = models.SingleParModel(conf)
                cb = [study.keras_callback(trial, objective_name='val_mae')]
                model.fit(train_ds, val_ds, cb)
                study.finalize(trial)
        else:
            train_ds, val_ds, test_ds = data.datasets(conf.input_dirs,
                                                      conf.img_shape)
            model = models.SingleParModel(conf)
            train_ds.cache()
            val_ds.cache()
            model.fit(train_ds, val_ds)
            model.evaluate(test_ds)


if __name__ == '__main__':
    main()
