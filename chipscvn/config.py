"""Provides configuration from .json files

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module produces a configuration namespace from an input .json config
file that can be used in the rest of the chips-cvn code. It also formats
the output directories for experiments.
"""

import json
from bunch import Bunch
import shutil
import os


def process_json(json_config):
    """Returns the configuration namespace specified in the config file."""
    with open(json_config, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Bunch(config_dict)  # Convert dict to namespace
    return config, config_dict


def create_directory(config):
    """Creates the directory for the experiment defined in the config."""
    # Need to force the location of the experiment dir relative to script
    experiments_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "../experiments")
    try:
        os.mkdir(experiments_dir)
    except FileExistsError:
        pass

    exp_dir = os.path.join(experiments_dir, config.experiment)
    if os.path.isdir(exp_dir) and config.type != "test":
        shutil.rmtree(exp_dir)

    try:
        os.mkdir(exp_dir)
    except FileExistsError:
        pass

    return exp_dir


def process_config(json_config):
    """Get the configuration and create experiment directories."""
    config, _ = process_json(json_config)
    config.exp_dir = create_directory(config)
    return config
