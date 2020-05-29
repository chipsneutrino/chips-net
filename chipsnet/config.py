# -*- coding: utf-8 -*-

"""Provides a configuration namespace from a yaml file

This module produces a configuration namespace from an input yaml config
file that can be used in the rest of the chipsnet code. It also formats
the output directories for experiments.
"""

import os
import shutil
import yaml
from dotmap import DotMap


def get(config_path):
    """Returns a configuration namespace generated from the config file.
    Args:
        config_path (str): Input configuration file path
    Returns:
        dotmap.DotMap: Dotmap configuration namespace
    """
    with open(config_path, 'r') as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.FullLoader)

    config_space = DotMap(config_dict)  # Convert yaml dict to namespace
    config_space.config_path = config_path

    return config_space


def setup_dirs(config, remove_old):
    """Sets up the experiment output directories.
    Args:
        config (dotmap.DotMap): Configuration namespace
        remove_old (bool): Should we remove old directories at the same path
    """
    config.exp.exp_dir = os.path.join(config.exp.output_dir, config.exp.name)
    if remove_old:
        shutil.rmtree(config.exp.exp_dir, ignore_errors=True)

    # Set the experiment directories
    os.makedirs(config.exp.output_dir, exist_ok=True)
    os.makedirs(config.exp.exp_dir, exist_ok=True)
    config.exp.tensorboard_dir = os.path.join(config.exp.exp_dir, 'tensorboard/')
    os.makedirs(config.exp.tensorboard_dir, exist_ok=True)
    config.exp.checkpoints_dir = os.path.join(config.exp.exp_dir, 'checkpoints/')
    os.makedirs(config.exp.checkpoints_dir, exist_ok=True)

    if remove_old:  # Copy file to keep record of training configuration
        shutil.copyfile(config.config_path, os.path.join(config.exp.exp_dir, "config.yaml"))
