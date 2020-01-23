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
import time
import sherpa
import ROOT
import config
import data
import models
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def run_study(config):
    """Creates a SHERPA hyperparameter study."""
    pars = models.get_model(config).study_parameters()
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=config.trials)
    study = sherpa.Study(parameters=pars, algorithm=algorithm, lower_is_better=False,
                         output_dir=config.exp_dir)

    for trial in study:
        for key in trial.parameters.keys():
            config[key] = trial.parameters[key]  # Adjust the configuration for this trial

        train_ds, val_ds = data.get_train_and_val_ds(config.input_dirs, config.img_shape)

        model = models.get_model(config)
        model.build()
        cb = [study.keras_callback(trial, objective_name='val_loss', context_names=['val_pdg_accuracy', 'val_type_accuracy', 'val_nuEnergy_mae'])]
        model.fit(train_ds, val_ds, cb)
        study.finalize(trial)
    study.save()


def train_model(config):
    """Trains a model according to the configuration."""
    train_ds, val_ds = data.get_train_and_val_ds(config.input_dirs, config.img_shape)
    model = models.get_model(config)
    model.build()
    model.fit(train_ds, val_ds)


def test_model(config):
    """Evaluate the trained model on the test dataset."""

    model = models.get_trained_model(config)
    h = ROOT.TH1F("test", "test", 100, -2000, 2000)
    test_ds = data.get_test_ds(config.input_dirs, config.img_shape)
    test_ds = test_ds.batch(64)

    start_time = time.time()
    for x, y in test_ds:
        predictions = model.model.predict(x)
        for i in range(len(predictions)):
            h.Fill(float(predictions[i] - y["lepEnergy"][i]))
    print("--- %s seconds ---" % (time.time() - start_time))

    myfile = ROOT.TFile("test.root", "RECREATE")
    h.Write()
    myfile.Close()


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description='CHIPS CVN')
    parser.add_argument('config', help='path to the configuration file')
    return parser.parse_args()


def gpu_setup():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def main():
    """Main function called by script."""
    print("\nCHIPS CVN - It's Magic\n")

    gpu_setup()

    args = parse_args()
    conf = config.process_config(args.config)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if conf.type == "study":
            run_study(conf)
        elif conf.type == "train":
            train_model(conf)
        elif conf.type == "test":
            test_model(conf)
        else:
            print("Error: type not valid!")
            raise SystemExit


if __name__ == '__main__':
    main()
