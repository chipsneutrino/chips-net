"""Utilities module

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module contains varous utility methods required throughout chipscvn.
This includes helper, plotting and output functions.
"""

import tensorflow as tf
import chipscvn.models as models
import chipscvn.trainers as trainers
import chipscvn.studies as studies
import chipscvn.evaluators as evaluators


def gpu_setup():
    """Sets up the system GPU's, memory growth is turned on."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print("--- ", len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs ---")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def get_model(config):
    """Returns the correct model for the configuration."""
    if config.model.name == "single_par":
        return models.SingleParModel(config)
    elif config.model.name == "classification":
        return models.ClassificationModel(config)
    elif config.model.name == "multi_task":
        return models.MultiTaskModel(config)
    else:
        print("Error: model not valid!")
        raise SystemExit


def get_trainer(config, model, data):
    """Returns the correct trainer for the configuration."""
    if config.model.name == "single_par":
        return trainers.BasicTrainer(config, model, data)
    elif config.model.name == "classification":
        return trainers.BasicTrainer(config, model, data)
    elif config.model.name == "multi_task":
        return trainers.BasicTrainer(config, model, data)
    else:
        print("Error: model not valid!")
        raise SystemExit


def get_study(config):
    """Returns the correct study for the configuration."""
    if config.model.name == "single_par":
        return studies.SingleParStudy(config)
    elif config.model.name == "classification":
        return studies.ClassificationStudy(config)
    elif config.model.name == "multi_task":
        return studies.MultiTaskStudy(config)
    else:
        print("Error: model not valid!")
        raise SystemExit


def get_evaluator(config):
    """Returns the correct evaluator for the configuration."""
    if config.model.name == "single_par":
        raise NotImplementedError
    elif config.model.name == "classification":
        return evaluators.ClassificationEvaluator(config)
    elif config.model.name == "multi_task":
        raise NotImplementedError
    else:
        print("Error: model not valid!")
        raise SystemExit
