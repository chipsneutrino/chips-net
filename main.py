# Runs the training of the various CVN networks
# Author: Josh Tingey
# Email: j.tingey.16@ucl.ac.uk

import argparse
import config
import data
import models
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description='CHIPS CVN')
    parser.add_argument('-i', '--input',  help='path to input directory')
    parser.add_argument('-o', '--output', help='Output .txt file')
    parser.add_argument('-c', '--config', help='Config .json file')
    return parser.parse_args()


def main():
    print("\nCHIPS CVN - It's Magic\n")
    args = parse_args()
    conf = config.process_config(args.config)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        train_ds = data.dataset(["/mnt/storage/jtingey/tf/train/"])
        val_ds = data.dataset(["/mnt/storage/jtingey/tf/val/"])
        test_ds = data.dataset(["/mnt/storage/jtingey/tf/test/"])

        model = models.PPEModel(conf)
        model.build()
        model.compile()
        model.plot()
        model.summary()
        model.fit(train_ds, val_ds)
        model.evaluate(test_ds)


if __name__ == '__main__':
    main()
