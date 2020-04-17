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
    elif config.model.name == "beam_all":
        return models.BeamAllModel(config)
    elif config.model.name == "beam_full_comb":
        return models.BeamFullCombModel(config)
    elif config.model.name == "beam_nu_nc_comb":
        return models.BeamNuNCCombModel(config)
    elif config.model.name == "beam_nc_comb":
        return models.BeamNCCombModel(config)
    elif config.model.name == "beam_multi":
        return models.BeamMultiModel(config)
    elif config.model.name == "beam_all_inception":
        return models.BeamAllInceptionModel(config)
    else:
        raise NotImplementedError


def get_trainer(config, model, data):
    """Returns the correct trainer for the configuration."""
    return trainers.BasicTrainer(config, model, data)


def get_study(config):
    """Returns the correct study for the configuration."""
    if config.model.name == "parameter":
        return studies.ParameterStudy(config)
    elif config.model.name == "cosmic":
        return studies.StandardStudy(config)
    elif config.model.name == "beam_all":
        return studies.StandardStudy(config)
    elif config.model.name == "beam_full_comb":
        return studies.StandardStudy(config)
    elif config.model.name == "beam_nu_nc_comb":
        return studies.StandardStudy(config)
    elif config.model.name == "beam_nc_comb":
        return studies.StandardStudy(config)
    elif config.model.name == "beam_multi":
        return studies.MultiStudy(config)
    elif config.model.name == "beam_all_inception":
        return studies.StandardStudy(config)
    else:
        raise NotImplementedError


def get_evaluator(config):
    """Returns the correct evaluator for the configuration."""
    if config.eval.type == "combined":
        return evaluators.CombinedEvaluator(config)
    if config.eval.type == "energy":
        return evaluators.EnergyEvaluator(config)
    else:
        raise NotImplementedError
