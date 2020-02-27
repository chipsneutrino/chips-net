"""Provides the CVN configuration from yaml files

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module produces a configuration namespace from an input yaml config
file that can be used in the rest of the chips-cvn code. It also formats
the output directories for experiments.
"""

import os
import shutil
import yaml
from dotmap import DotMap


def get(config_path):
    """Returns the configuration namespace specified in the config file."""
    with open(config_path, 'r') as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
        
    return DotMap(config_dict)  # Convert yaml dict to namespace


def setup_dirs(config, remove_first):
    """Sets up the experiment output directories."""
    config.exp.exp_dir = os.path.join(config.exp.experiment_dir, config.exp.name)
    if remove_first:
        shutil.rmtree(config.exp.exp_dir, ignore_errors=True)

    # Set the experiment directories
    os.makedirs(config.exp.exp_dir, exist_ok=True)
    config.exp.tensorboard_dir = os.path.join(config.exp.exp_dir, 'tensorboard')
    os.makedirs(config.exp.tensorboard_dir, exist_ok=True)
    config.exp.checkpoints_dir = os.path.join(config.exp.exp_dir, 'checkpoints')
    os.makedirs(config.exp.checkpoints_dir, exist_ok=True)

