"""Main training acript

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This script is the main chips-cvn training script. It trains a given
model, with the specified input data and evaluates the test dataset on
the output.
"""

import argparse
import config
import data
import models
import tensorflow as tf
import logging


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description='CHIPS CVN')
    parser.add_argument('config', help='path to the configuration file')
    return parser.parse_args()


def main():
    """Main function called by script."""
    print("\nCHIPS CVN - It's Magic\n")
    conf = config.process_config(parse_args().config)

    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        train_ds, val_ds, test_ds = data.datasets(conf.input_dirs,
                                                  conf.img_shape)

        model = models.SingleParModel(conf)
        model.build()
        model.compile()
        model.fit(train_ds, val_ds)
        model.evaluate(test_ds)


if __name__ == '__main__':
    main()
