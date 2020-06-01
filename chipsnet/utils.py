# -*- coding: utf-8 -*-

"""Utility module containing lots of helpful methods for evaluation and plotting
"""

import time
import copy

import pandas as pd
import numpy as np
from tensorflow.keras import Model
from tqdm import tqdm

import chipsnet.config
import chipsnet.data as data
import chipsnet.models


def data_from_conf(config, name):
    """Get the input data from the configuration.
    Args:
        config (dotmap.DotMap): Base configuration namespace
        name (str): Data name
    Returns:
        chipsnet.data reader: Data reader from configuration
    """
    data_config = copy.copy(config)
    data_config.data = config.samples[name]
    data = chipsnet.data.Reader(data_config)
    return data


def model_from_conf(config, name):
    """Get and load the model from the configuration.
    Args:
        config (dotmap.DotMap): Base configuration namespace
        name (str): Model name
    Returns:
        chipsnet.models model: Model from the configuration
    """
    model_config = copy.copy(config)
    model_config.model = config.models[name]
    model_config.exp.output_dir = config.models[name].dir
    model_config.exp.name = config.models[name].path
    chipsnet.config.setup_dirs(model_config, False)
    model = chipsnet.models.get_model(model_config)
    model.load()
    return model


def run_inference(data, events, model, num_events, prefix=''):
    """Get model outputs for test data.
    """
    data = data.testing_ds.take(num_events)
    data = data.batch(64, drop_remainder=True)  # Batch to make inference faster
    num_model_outputs = len(model.config.model.labels)
    outputs = model.model.predict(data)
    for i, label in enumerate(model.config.model.labels):
        base_key = prefix + '_pred_' + label
        if int(num_model_outputs) == 1:
            events[base_key] = outputs
        elif outputs[i].shape[1] == 1:
            events[base_key] = outputs[i]
        else:
            for j in range(outputs[i].shape[1]):
                key = base_key + '_' + str(j)
                events[key] = outputs[i][:, j]
    return events


def apply_weights(events, total_num, nuel_frac, anuel_frac, numu_frac,
                  anumu_frac, cosmic_frac):
    """Calculate the weights to apply categorically.
    """
    tot_nuel = events[(events[data.MAP_NU_TYPE.name] == 0) &
                      (events[data.MAP_SIGN_TYPE.name] == 0) &
                      (events[data.MAP_COSMIC_CAT.name] == 0)].shape[0]
    tot_anuel = events[(events[data.MAP_NU_TYPE.name] == 0) &
                       (events[data.MAP_SIGN_TYPE.name] == 1) &
                       (events[data.MAP_COSMIC_CAT.name] == 0)].shape[0]
    tot_numu = events[(events[data.MAP_NU_TYPE.name] == 1) &
                      (events[data.MAP_SIGN_TYPE.name] == 0) &
                      (events[data.MAP_COSMIC_CAT.name] == 0)].shape[0]
    tot_anumu = events[(events[data.MAP_NU_TYPE.name] == 1) &
                       (events[data.MAP_SIGN_TYPE.name] == 1) &
                       (events[data.MAP_COSMIC_CAT.name] == 0)].shape[0]
    tot_cosmic = events[events[data.MAP_COSMIC_CAT.name] == 1].shape[0]

    nuel_weight = (1.0/tot_nuel)*(nuel_frac * total_num)
    anuel_weight = (1.0/tot_anuel)*(anuel_frac * total_num)
    numu_weight = (1.0/tot_numu)*(numu_frac * total_num)
    anumu_weight = (1.0/tot_anumu)*(anumu_frac * total_num)
    cosmic_weight = (1.0/tot_cosmic)*(cosmic_frac * total_num)

    events['weight'] = events.apply(
        add_weight,
        axis=1,
        args=(nuel_weight, anuel_weight, numu_weight, anumu_weight, cosmic_weight)
    )

    print("el:{:.4f}({}), ael:{:.4f}({}), mu:{:.4f}({}), amu:{:.4f}({}), cos:{:.4f}({})".format(
        nuel_weight, tot_nuel,
        anuel_weight, tot_anuel,
        numu_weight, tot_numu,
        anumu_weight, tot_anumu,
        cosmic_weight, tot_cosmic)
    )
    return events


def add_weight(event, nuel_weight, anuel_weight, numu_weight, anumu_weight, cosmic_weight):
    """Add the correct weight to each event.
    Args:
        event (dict): Pandas event(row) dict
    Returns:
        float: Weight to use for event
    """
    if (event[data.MAP_NU_TYPE.name] == 0 and
            event[data.MAP_SIGN_TYPE.name] == 0 and
            event[data.MAP_COSMIC_CAT.name] == 0):
        return nuel_weight
    elif (event[data.MAP_NU_TYPE.name] == 0 and
            event[data.MAP_SIGN_TYPE.name] == 1 and
            event[data.MAP_COSMIC_CAT.name] == 0):
        return anuel_weight
    elif (event[data.MAP_NU_TYPE.name] == 1 and
            event[data.MAP_SIGN_TYPE.name] == 0 and
            event[data.MAP_COSMIC_CAT.name] == 0):
        return numu_weight
    elif (event[data.MAP_NU_TYPE.name] == 1 and
            event[data.MAP_SIGN_TYPE.name] == 1 and
            event[data.MAP_COSMIC_CAT.name] == 0):
        return anumu_weight
    elif (event[data.MAP_COSMIC_CAT.name] == 1):
        return cosmic_weight
    else:
        raise NotImplementedError


def cut_apply(variable, value, type='greater_than'):
    def cut_func(event):
        if type == 'greater_than':
            return (event[variable] >= value)
        elif type == 'lower_than':
            return (event[variable] <= value)
        else:
            raise NotImplementedError
    return cut_func


def cut_summary(events, variables):
    """Print how each category is affected by the cut.
    """
    for i in range(len(data.MAP_FULL_COMB_CAT.labels)):
        cat_events = events[events[data.MAP_FULL_COMB_CAT.name] == i]
        survived_events = None
        for var in variables:
            survived_events = cat_events[cat_events[var] == 0]
        print("{}-> Total {}, Survived: {}\n".format(
            data.MAP_FULL_COMB_CAT.labels[i],
            cat_events.shape[0],
            survived_events.shape[0]/cat_events.shape[0])
        )


