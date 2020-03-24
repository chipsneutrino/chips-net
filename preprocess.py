"""Data preprocessing script

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This script is the main chips-cvn training script. Given the input
configuration it trains the given model and then evaluates the test
dataset. It can also carry out hyperparameter optimisation using
SHERPA which requires a modified configuration file.
"""

import argparse
import chipscvn.data as data


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description='CHIPS CVN Data')
    parser.add_argument('directory', help='path to input directory')
    parser.add_argument('-s', '--split', help='val and test split fraction', default=0.1)
    parser.add_argument('-g', '--geom', help='detector geometry name', default='chips_1200_sk1pe')
    parser.add_argument('-j', '--join', help='how many input files to join together', default=10)
    parser.add_argument('--all', action='store_true',
                        help='pass through all maps to tfrecords file')
    parser.add_argument('--single', action='store_true', help='Use a single process')
    return parser.parse_args()


def main():
    """Main function called by script."""
    args = parse_args()
    creator = data.DataCreator(
        args.directory, args.geom, args.split, args.join, args.single, args.all)
    creator.preprocess()


if __name__ == '__main__':
    main()
