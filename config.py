# Reads the .json configuration
# Author: Josh Tingey
# Email: j.tingey.16@ucl.ac.uk

import json
from bunch import Bunch
import os

def process_json(json_config):
    # parse the configurations from the config json file provided
    with open(json_config, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict

def create_directories(config):
    try:
        os.mkdir("experiments")
    except:
        pass
    try:
        os.mkdir(os.path.join("experiments", config.exp_name))
    except:
        pass
    try:
        os.mkdir(os.path.join("experiments", config.exp_name, "summary/"))
    except:
        pass
    try:
        os.mkdir(os.path.join("experiments", config.exp_name, "checkpoint/"))
    except:
        pass

def process_config(json_config):
    config, _ = process_json(json_config)
    config.summary_dir = os.path.join("experiments", config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoint/")
    create_directories(config)
    return config

def get_config(json_config):
    config, _ = process_json(json_config)    
    return config