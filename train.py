"""Main training acript

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This script is the main chips-cvn training script. It trains a given
model, with the specified input data and evaluates the test dataset on
the output.
"""

import os.path as path
import argparse
import config
import data
import models
import tensorflow as tf


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description='CHIPS CVN')
    parser.add_argument('-i', '--input',  help='path to input directory')
    parser.add_argument('-o', '--output', help='Output .txt file')
    parser.add_argument('-c', '--config', help='Config .json file')
    return parser.parse_args()


def main():
    """Main function called by script."""
    print("\nCHIPS CVN - It's Magic\n")
    args = parse_args()
    conf = config.process_config(args.config)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        train_ds = data.dataset([path.join(args.input, "train")])
        val_ds = data.dataset([path.join(args.input, "val")])
        test_ds = data.dataset([path.join(args.input, "test")])

        model = models.PPEModel(conf)
        model.build()
        model.compile()
        model.plot()
        model.summary()
        model.fit(train_ds, val_ds)
        model.evaluate(test_ds)


if __name__ == '__main__':
    main()
