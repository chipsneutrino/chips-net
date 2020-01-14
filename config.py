"""Provides configuration from .json files

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module produces a configuration namespace from an input .json config
file that can be used in the rest of the chips-cvn code. It also formats
the output directories for experiments.
"""

import json
from bunch import Bunch
import os


def process_json(json_config):
    """Returns the configuration namespace specified in the config file."""
    with open(json_config, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Bunch(config_dict)  # Convert dict to namespace
    return config, config_dict


def create_directories(config):
    """Creates the directories for the experiment defined in the config."""
    try:
        os.mkdir("experiments")
        os.mkdir(os.path.join("experiments", config.exp_name))
        os.mkdir(os.path.join("experiments", config.exp_name, "summary/"))
        os.mkdir(os.path.join("experiments", config.exp_name, "checkpoint/"))
    except FileExistsError:
        pass


def process_config(json_config):
    """Get the configuration and create experiment directories."""
    config, _ = process_json(json_config)
    config.summary_dir = os.path.join("experiments", config.exp_name,
                                      "summary/")
    config.checkpoint_dir = os.path.join("experiments", config.exp_name,
                                         "checkpoint/")
    create_directories(config)
    return config


def get_config(json_config):
    """Get the configuration without creating experiment directories."""
    config, _ = process_json(json_config)
    return config
