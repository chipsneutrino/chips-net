"""Provides the CVN configuration from yaml files

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module produces a configuration namespace from an input yaml config
file that can be used in the rest of the chips-cvn code. It also formats
the output directories for experiments.
"""

import os
import yaml
from dotmap import DotMap


def process_yaml(config_path):
    """Returns the configuration namespace specified in the config file."""
    with open(config_path, 'r') as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
        
    return DotMap(config_dict)  # Convert yaml dict to namespace


def process_config(config_path):
    """Setup the config namespace and experiment output directories."""
    config = process_yaml(config_path)

    # Set the experiment directories
    config.exp.exp_dir = os.path.join(config.exp.experiment_dir, config.exp.name)
    os.makedirs(config.exp.exp_dir, exist_ok=True)
    config.exp.tensorboard_dir = os.path.join(config.exp.exp_dir, 'tensorboard')
    os.makedirs(config.exp.tensorboard_dir, exist_ok=True)
    config.exp.checkpoints_dir = os.path.join(config.exp.exp_dir, 'checkpoints')
    os.makedirs(config.exp.checkpoints_dir, exist_ok=True)

    return config
