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
    if config.model.name == "parameter":
        return models.ParameterModel(config)
    elif config.model.name == "cosmic":
        return models.CosmicModel(config)
    elif config.model.name == "beam":
        return models.BeamModel(config)
    elif config.model.name == "combined":
        return models.CombinedCatModel(config)
    elif config.model.name == "multi":
        return models.MultiModel(config)
    else:
        print("Error: model not valid!")
        raise SystemExit


def get_trainer(config, model, data):
    """Returns the correct trainer for the configuration."""
    if config.model.name == "parameter":
        return trainers.BasicTrainer(config, model, data)
    elif config.model.name == "cosmic":
        return trainers.BasicTrainer(config, model, data)
    elif config.model.name == "beam":
        return trainers.BasicTrainer(config, model, data)
    elif config.model.name == "combined":
        return trainers.BasicTrainer(config, model, data)
    elif config.model.name == "multi":
        return trainers.BasicTrainer(config, model, data)
    else:
        print("Error: trainer not valid!")
        raise SystemExit


def get_study(config):
    """Returns the correct study for the configuration."""
    if config.model.name == "parameter":
        return studies.ParameterStudy(config)
    elif config.model.name == "cosmic":
        return studies.CosmicStudy(config)
    elif config.model.name == "beam":
        return studies.BeamStudy(config)
    elif config.model.name == "combined":
        return studies.CombinedStudy(config)
    elif config.model.name == "multi":
        return studies.MultiStudy(config)
    else:
        print("Error: study not valid!")
        raise SystemExit


def get_evaluator(config):
    """Returns the correct evaluator for the configuration."""
    if config.eval.type == "combined":
        return evaluators.CombinedEvaluator(config)
    else:
        print("Error: evaluator not valid!")
        raise SystemExit
