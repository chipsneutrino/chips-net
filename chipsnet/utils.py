# -*- coding: utf-8 -*-

"""Utility module containing lots of helpful methods for evaluation and plotting."""

import os
import time
import copy
import ast
import math

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from tf_explain.core.activations import ExtractActivations
from tf_explain.core.vanilla_gradients import VanillaGradients
from tf_explain.core.smoothgrad import SmoothGrad
from tf_explain.core.integrated_gradients import IntegratedGradients
from tf_explain.core.gradients_inputs import GradientsInputs
from scipy.optimize import curve_fit
import uproot
from pandarallel import pandarallel

import chipsnet.config
import chipsnet.data as data
import chipsnet.models


pandarallel.initialize(verbose=1)


def data_from_conf(config, name):
    """Get the named data.Reader as defined in the configuration.

    Args:
        config (dotmap.DotMap): base configuration namespace
        name (str): data name

    Returns:
        chipsnet.data.Reader: data reader from configuration
    """
    # We copy the configuration so we don't modify it
    data_config = copy.copy(config)
    data_config.data.input_dirs = config.samples[name].input_dirs
    data_config.data.channels = config.samples[name].channels
    if config.samples[name].seperate_channels is False:
        data_config.data.seperate_channels = False
    if config.samples[name].augment is True:
        data_config.data.augment = True
        data_config.data.aug_factor_mean = config.samples[name].aug_factor_mean
        data_config.data.aug_factor_sigma = config.samples[name].aug_factor_sigma
        data_config.data.aug_abs_mean = config.samples[name].aug_abs_mean
        data_config.data.aug_abs_sigma = config.samples[name].aug_abs_sigma
        data_config.data.aug_noise_mean = config.samples[name].aug_noise_mean
        data_config.data.aug_noise_sigma = config.samples[name].aug_noise_sigma
    data = chipsnet.data.Reader(data_config)
    return data


def model_from_conf(config, name, checkpoint="best"):
    """Get the loaded models.Model corresponding to the given name and configuration.

    Args:
        config (dotmap.DotMap): base configuration namespace
        name (str): model name
        checkpoint (str): which checkpoint to load

    Returns:
        chipsnet.models.Model: model loaded from the configuration
    """
    # We copy the configuration so we don't modify it
    model_config = copy.copy(config)
    model_config.model.type = config.models[name].type
    model_config.model.labels = config.models[name].labels
    model_config.exp.name = config.models[name].path
    model_config.data.channels = config.models[name].channels
    if config.models[name].seperate_channels is False:
        model_config.data.seperate_channels = False
    if config.models[name].reco_pars is False:
        model_config.model.reco_pars = False
    if config.models[name].precision_policy == "float32":
        model_config.model.precision_policy = "float32"
    chipsnet.config.setup_dirs(model_config, False)
    model = chipsnet.models.Model(model_config)
    model.load(checkpoint)
    return model


def model_history(config, name):
    """Get the saved model training history.

    Args:
        config (dotmap.DotMap): configuration namespace
        name (str): model name

    Returns:
        pd.DataFrame: dataframe containing model training history
    """
    model_config = copy.copy(config)
    model_config.model = config.models[name]
    model_config.exp.name = config.models[name].path
    model_config.data.channels = config.models[name].channels
    chipsnet.config.setup_dirs(model_config, False)
    history = pd.read_csv(os.path.join(model_config.exp.exp_dir, "history.csv"))
    history_dict = {}
    for key in history.keys():
        history_dict[key] = ast.literal_eval(history[key].values[0])
    return pd.DataFrame.from_dict(history_dict)


def evaluate(
    config,
    data_name,
    m_names=[],
    m_cats=["t_all_cat"],
    just_out=False,
    just_cosmic=False,
    checkpoint="best",
    exclude_images=True,
):
    """Fully process a dataset through a list of models and run a standard evaluation.

    Args:
        config (dotmap.DotMap): configuration namespace
        data_name (str): data name
        model_names (List[str]): model names
        verbose (bool): should we print summaries?
        just_out (bool): just return the outputs not the events?
        checkpoint (str): which checkpoint to load
        exclude_images (bool): don't include the input images

    Returns:
        pd.DataFrame: fully processed events dataframe
    """
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        print(
            "\n************************ Evaluating {} ************************".format(
                data_name
            )
        )
        start_time = time.time()

        # Get the dataframe from the dataset name
        testing_data = data_from_conf(config, data_name)
        ev = testing_data.test_val_df(config.eval.examples, exclude_images)

        # Run all the required inference
        for m_name in m_names:
            model = model_from_conf(config, m_name, checkpoint)
            inf_out = run_inference(
                testing_data.test_val_ds(config.eval.examples, batch_size=256),
                model,
                prefix=m_name + "_",
            )

            ev = pd.concat([ev, inf_out], axis=1)

            if "t_nu_energy" in model.config.model.labels:
                ev[m_name + "_frac_nu_energy"] = (
                    ev[m_name + "_pred_t_nu_energy"] - ev["t_nu_energy"]
                ) / ev["t_nu_energy"]

            if "t_lep_energy" in model.config.model.labels:
                ev[m_name + "_frac_lep_energy"] = (
                    ev[m_name + "_pred_t_lep_energy"] - ev["t_lep_energy"]
                ) / ev["t_lep_energy"]

            if "t_had_energy" in model.config.model.labels:
                ev[m_name + "_frac_had_energy"] = (
                    ev[m_name + "_pred_t_had_energy"] - ev["t_had_energy"]
                ) / ev["t_had_energy"]

        # Calculate the fraction of nu energy in lepton
        ev["frac_energy"] = ev["t_lep_energy"] / ev["t_nu_energy"]

        # Apply the event weights
        ev = apply_weights(
            ev,
            total_num=config.eval.weights.total,
            nuel_frac=config.eval.weights.nuel,
            anuel_frac=config.eval.weights.anuel,
            numu_frac=config.eval.weights.numu,
            anumu_frac=config.eval.weights.anumu,
            cosmic_frac=config.eval.weights.cosmic,
            osc_file_name=config.eval.weights.osc_file_name,
            verbose=True,
        )

        # Apply the standard cuts
        ev = apply_standard_cuts(
            ev,
            cosmic_cut=config.eval.cuts.cosmic,
            q_cut=config.eval.cuts.q,
            h_cut=config.eval.cuts.h,
            theta_cut=config.eval.cuts.theta,
            phi_cut=config.eval.cuts.phi,
            verbose=True,
        )

        # Classify into fully combined categories and print the classification reports
        outputs = []

        for m_name, m_cat in zip(m_names, m_cats):
            output = {}
            if m_cat in ["t_cosmic_cat", "energy"]:
                continue

            # Combine categories into fully combined ones
            ev = full_comb_combine(ev, m_cat, prefix=m_name + "_")

            # Run classification and print report
            class_prefix = m_name + "_" + "pred_t_comb_cat_"
            ev[m_name + "_comb_cat_class"] = ev.apply(
                classify, axis=1, args=(3, class_prefix)
            )
            class_prefix = m_name + "_" + "pred_" + m_cat + "_"
            ev[m_name + "_" + m_cat + "_class"] = ev.apply(
                classify,
                axis=1,
                args=(chipsnet.data.get_map(m_cat)["categories"], class_prefix),
            )

            # Combined classification report and confusion matrix
            comb_cats = chipsnet.data.get_map("t_comb_cat")["categories"] + 1
            comb_labels = chipsnet.data.get_map("t_comb_cat")["labels"]
            if len(ev[ev["t_cosmic_cat"] == 1]) == 0:
                comb_cats = comb_cats - 1
                comb_labels = comb_labels[:comb_cats]
            comb_report = classification_report(
                ev["t_comb_cat"],
                ev[m_name + "_comb_cat_class"],
                labels=[x for x in range(comb_cats)],
                target_names=comb_labels,
                sample_weight=ev["w"],
                zero_division=0,
                output_dict=True,
            )
            comb_matrix = confusion_matrix(
                ev["t_comb_cat"],
                ev[m_name + "_comb_cat_class"],
                labels=[x for x in range(comb_cats)],
                sample_weight=ev["w"],
                normalize="true",
            )
            comb_matrix = np.rot90(comb_matrix, 1)
            comb_matrix = pd.DataFrame(comb_matrix)
            output["comb_report"] = comb_report
            output["comb_matrix"] = comb_matrix

            # Model category classification report and confusion matrix
            cat_cats = chipsnet.data.get_map(m_cat)["categories"] + 1
            cat_labels = chipsnet.data.get_map(m_cat)["labels"]
            if len(ev[ev["t_cosmic_cat"] == 1]) == 0:
                cat_cats = cat_cats - 1
                cat_labels = cat_labels[:cat_cats]
            cat_report = classification_report(
                ev[m_cat],
                ev[m_name + "_" + m_cat + "_class"],
                labels=[x for x in range(cat_cats)],
                target_names=cat_labels,
                sample_weight=ev["w"],
                zero_division=0,
                output_dict=True,
            )
            cat_matrix = confusion_matrix(
                ev[m_cat],
                ev[m_name + "_" + m_cat + "_class"],
                labels=[x for x in range(cat_cats)],
                sample_weight=ev["w"],
                normalize="true",
            )
            cat_matrix = np.rot90(cat_matrix, 1)
            cat_matrix = pd.DataFrame(cat_matrix)
            output["cat_report"] = cat_report
            output["cat_matrix"] = cat_matrix

            curves_output = calculate_curves(ev, prefix=m_name + "_")
            eff_output = calculate_eff_pur(
                ev, curves_output["max_fom_cuts_0"], prefix=m_name + "_"
            )
            output["cuts"] = curves_output["cuts"]
            output["sig_effs"] = curves_output["sig_effs"]
            output["bkg_effs"] = curves_output["bkg_effs"]
            output["purs"] = curves_output["purs"]
            output["foms_0"] = curves_output["foms_0"]
            output["foms_1"] = curves_output["foms_1"]
            output["fom_effs"] = eff_output["fom_effs"]
            output["fom_purs"] = eff_output["fom_purs"]
            output["sig_effs_auc"] = curves_output["sig_effs_auc"]
            output["bkg_effs_auc"] = curves_output["bkg_effs_auc"]
            output["pur_auc"] = curves_output["pur_auc"]
            output["fom_auc_0"] = curves_output["fom_auc_0"]
            output["fom_auc_1"] = curves_output["fom_auc_1"]
            output["roc_auc"] = curves_output["roc_auc"]
            output["prc_auc"] = curves_output["prc_auc"]
            output["max_foms_0"] = curves_output["max_foms_0"]
            output["max_foms_1"] = curves_output["max_foms_1"]
            output["max_fom_cuts_0"] = curves_output["max_fom_cuts_0"]
            output["max_fom_cuts_1"] = curves_output["max_fom_cuts_1"]
            output["max_fom_passed_0"] = curves_output["max_fom_passed_0"]
            output["max_fom_passed_1"] = curves_output["max_fom_passed_1"]
            outputs.append(output)

            if just_cosmic:
                continue

            print(
                "\n------------------------ {} report ------------------------".format(
                    m_name
                )
            )
            print(
                "- Comb-> Prec: ({:.5f},{:.5f}), Rec: ({:.5f},{:.5f}), F1: ({:.5f},{:.5f})".format(
                    comb_report["weighted avg"]["precision"],
                    comb_report["macro avg"]["precision"],
                    comb_report["weighted avg"]["recall"],
                    comb_report["macro avg"]["recall"],
                    comb_report["weighted avg"]["f1-score"],
                    comb_report["macro avg"]["f1-score"],
                )
            )
            print(
                "- Cat->  Prec: ({:.5f},{:.5f}), Rec: ({:.5f},{:.5f}), F1: ({:.5f},{:.5f})".format(
                    cat_report["weighted avg"]["precision"],
                    cat_report["macro avg"]["precision"],
                    cat_report["weighted avg"]["recall"],
                    cat_report["macro avg"]["recall"],
                    cat_report["weighted avg"]["f1-score"],
                    cat_report["macro avg"]["f1-score"],
                )
            )
            print(
                "\n- Nuel-> ROC-AUC: {:.5f}, PRC-AUC: {:.5f}, S-Eff: {:.5f}, S-Pur: {:.5f}".format(
                    output["roc_auc"][0],
                    output["prc_auc"][0],
                    output["sig_effs"][0][np.where(output["cuts"] == 0.5)[0]][0],
                    output["purs"][0][np.where(output["cuts"] == 0.5)[0]][0],
                )
            )
            print(
                "- FOM1-> {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(
                    output["max_foms_0"][0],
                    output["max_fom_cuts_0"][0],
                    output["max_fom_passed_0"][0][0],
                    output["max_fom_passed_0"][0][1],
                    output["max_fom_passed_0"][0][2],
                    # output["max_fom_passed_0"][0][3],
                    output["sig_effs"][0][
                        np.where(output["cuts"] == output["max_fom_cuts_0"][0])[0]
                    ][0],
                    output["purs"][0][
                        np.where(output["cuts"] == output["max_fom_cuts_0"][0])[0]
                    ][0],
                )
            )

            print(
                "- FOM2-> {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(
                    output["max_foms_1"][0],
                    output["max_fom_cuts_1"][0],
                    output["max_fom_passed_1"][0][0],
                    output["max_fom_passed_1"][0][1],
                    output["max_fom_passed_1"][0][2],
                    # output["max_fom_passed_1"][0][3],
                    output["sig_effs"][0][
                        np.where(output["cuts"] == output["max_fom_cuts_1"][0])[0]
                    ][0],
                    output["purs"][0][
                        np.where(output["cuts"] == output["max_fom_cuts_1"][0])[0]
                    ][0],
                )
            )

            print(
                "\n- Numu-> ROC-AUC: {:.5f}, PRC-AUC: {:.5f}, S-Eff: {:.5f}, S-Pur: {:.5f}".format(
                    output["roc_auc"][1],
                    output["prc_auc"][1],
                    output["sig_effs"][1][np.where(output["cuts"] == 0.5)[0]][0],
                    output["purs"][1][np.where(output["cuts"] == 0.5)[0]][0],
                )
            )
            print(
                "- FOM1-> {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(
                    output["max_foms_0"][1],
                    output["max_fom_cuts_0"][1],
                    output["max_fom_passed_0"][1][0],
                    output["max_fom_passed_0"][1][1],
                    output["max_fom_passed_0"][1][2],
                    # output["max_fom_passed_0"][1][3],
                    output["sig_effs"][1][
                        np.where(output["cuts"] == output["max_fom_cuts_0"][1])[0]
                    ][0],
                    output["purs"][1][
                        np.where(output["cuts"] == output["max_fom_cuts_0"][1])[0]
                    ][0],
                )
            )

            print(
                "- FOM2-> {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n".format(
                    output["max_foms_1"][1],
                    output["max_fom_cuts_1"][1],
                    output["max_fom_passed_1"][1][0],
                    output["max_fom_passed_1"][1][1],
                    output["max_fom_passed_1"][1][2],
                    # output["max_fom_passed_1"][1][3],
                    output["sig_effs"][1][
                        np.where(output["cuts"] == output["max_fom_cuts_1"][1])[0]
                    ][0],
                    output["purs"][1][
                        np.where(output["cuts"] == output["max_fom_cuts_1"][1])[0]
                    ][0],
                )
            )

        print("took {:.2f} seconds".format(time.time() - start_time))

    # Return everything
    if just_out:
        return outputs
    else:
        return ev, outputs


def run_inference(ds, model, prefix=""):
    """Run predictions on the input dataset and append outputs to events dataframe.

    Args:
        ds (tf.dataset): data set to use for inference
        model (chipsnet.Model): model to use for prediction
        prefix (str): prefix to append to column name

    Returns:
        pd.DataFrame: events dataframe with model predictions
    """
    # Make the predictions
    outputs = model.model.predict(ds)

    # Append the predictions to the events dataframe
    events = pd.DataFrame()
    model_outputs = model.model.outputs
    for i, model_output in enumerate(model_outputs):
        base_key = prefix + "pred_" + model_output.name.split("/", 1)[0]
        for j in range(10):
            base_key = base_key.split("_" + str(j), 1)[0]

        output = None
        if len(model_outputs) == 1:
            output = outputs
        else:
            output = outputs[i]

        if output.shape[1] == 1:
            events[base_key] = output.flatten()
        else:
            for cat in range(output.shape[1]):
                key = base_key + "_" + str(cat)
                events[key] = output[:, cat]

    return events


def full_comb_combine(events, map_type, prefix=""):
    """Combine output scores into t_comb_cat scores.

    Args:
        events (pd.DataFrame): events dataframe to calculate combined category scores
        prefix (str): prefix to apply to model output values

    Returns:
        pd.DataFrame: events dataframe with combined category scores
    """

    def all_cat_apply(event, apply_prefix):
        nuel_cc_cats = [0, 1, 2, 3, 4, 5]
        nuel_cc_value = 0.0
        for cat in nuel_cc_cats:
            nuel_cc_value = nuel_cc_value + event[str(apply_prefix) + str(cat)]

        numu_cc_cats = [6, 7, 8, 9, 10, 11]
        numu_cc_value = 0.0
        for cat in numu_cc_cats:
            numu_cc_value = numu_cc_value + event[apply_prefix + str(cat)]

        nc_cats = [12, 13, 14, 15]
        nc_value = 0.0
        for cat in nc_cats:
            nc_value = nc_value + event[apply_prefix + str(cat)]

        return nuel_cc_value, numu_cc_value, nc_value

    def nc_comb_cat_apply(event, apply_prefix):
        nuel_cc_cats = [0, 1, 2, 3, 4, 5]
        nuel_cc_value = 0.0
        for cat in nuel_cc_cats:
            nuel_cc_value = nuel_cc_value + event[str(apply_prefix) + str(cat)]

        numu_cc_cats = [6, 7, 8, 9, 10, 11]
        numu_cc_value = 0.0
        for cat in numu_cc_cats:
            numu_cc_value = numu_cc_value + event[apply_prefix + str(cat)]

        nc_cats = [12]
        nc_value = 0.0
        for cat in nc_cats:
            nc_value = nc_value + event[apply_prefix + str(cat)]

        return nuel_cc_value, numu_cc_value, nc_value

    comb_prefix = prefix + "pred_t_comb_cat_"
    if map_type == "t_comb_cat":
        return events
    elif map_type == "t_all_cat":
        apply_prefix = prefix + "pred_t_all_cat_"
        events["scores"] = events.apply(all_cat_apply, axis=1, args=(apply_prefix,))
    elif map_type == "t_nc_comb_cat":
        apply_prefix = prefix + "pred_t_nc_comb_cat_"
        events["scores"] = events.apply(nc_comb_cat_apply, axis=1, args=(apply_prefix,))

    events[comb_prefix + "0"] = events.scores.parallel_map(lambda x: x[0])
    events[comb_prefix + "1"] = events.scores.parallel_map(lambda x: x[1])
    events[comb_prefix + "2"] = events.scores.parallel_map(lambda x: x[2])
    events.drop(["scores"], axis=1)
    return events


def apply_weights(
    events,
    total_num=1214165.85244438,  # for chips_1200
    nuel_frac=0.00003202064566,  # for chips_1200
    anuel_frac=0.00000208200747,  # for chips_1200
    numu_frac=0.00276174709613,  # for chips_1200
    anumu_frac=0.00006042213136,  # for chips_1200
    cosmic_frac=0.99714372811940,  # for chips_1200
    osc_file_name="./data/input/oscillations/matter_osc_cp_zero.root",
    verbose=False,
):
    """Calculate and apply the 'weight' column to scale events to predicted numbers.

    Args:
        events (pd.DataFrame): events dataframe to append weights to
        total_num (float): total number of expected events in a year
        nuel_frac (float): fraction of events from nuel
        anuel_frac (float): fraction of events from anuel
        numu_frac (float): fraction of events from numu
        anumu_frac (float): fraction of events from anumu
        cosmic_frac (float): fractions of events from cosmics
        osc_file_name (str): Oscillation data file name
        verbose (bool): should we print the weight summary?

    Returns:
        pd.DataFrame: events dataframe with weights
    """

    def apply_scale_weight(event, w_nuel, w_anuel, w_numu, w_anumu, w_cosmic):
        """Add the correct weight to each event.

        Args:
            event (dict): pandas event(row) dict
            w_nuel: nuel weight
            w_anuel: anuel weight
            w_numu: numu weight
            w_anumu: anumu weight
            w_cosmic: cosmic weight
        """
        if (
            event[data.MAP_NU_TYPE["name"]] == 0
            and event[data.MAP_SIGN_TYPE["name"]] == 0
            and event[data.MAP_COSMIC_CAT["name"]] == 0
            and event["t_sample_type"] == 0
        ):  # Beam nuel
            return w_nuel
        elif (
            event[data.MAP_NU_TYPE["name"]] == 0
            and event[data.MAP_SIGN_TYPE["name"]] == 0
            and event[data.MAP_COSMIC_CAT["name"]] == 0
            and event["t_sample_type"] == 1
        ):  # Appeared nuel
            return 1
        elif (
            event[data.MAP_NU_TYPE["name"]] == 0
            and event[data.MAP_SIGN_TYPE["name"]] == 1
            and event[data.MAP_COSMIC_CAT["name"]] == 0
            and event["t_sample_type"] == 0
        ):  # Beam anuel
            return w_anuel
        elif (
            event[data.MAP_NU_TYPE["name"]] == 1
            and event[data.MAP_SIGN_TYPE["name"]] == 0
            and event[data.MAP_COSMIC_CAT["name"]] == 0
            and event["t_sample_type"] == 0
        ):  # Beam numu
            return w_numu
        elif (
            event[data.MAP_NU_TYPE["name"]] == 1
            and event[data.MAP_SIGN_TYPE["name"]] == 1
            and event[data.MAP_COSMIC_CAT["name"]] == 0
            and event["t_sample_type"] == 0
        ):  # Beam anumu
            return w_anumu
        elif event[data.MAP_COSMIC_CAT["name"]] == 1 and event["t_sample_type"] == 2:
            return w_cosmic
        else:
            return 0

    def apply_osc_weight(event, numu_survival_prob, nuel_osc):
        """Add the correct weight to each event.

        Args:
            event (dict): pandas event(row) dict
            numu_survival_prob (np.array): numu survival probability array
            nuel_osc (np.array): numu appearance scaled probability array
        """
        if (
            event[data.MAP_NU_TYPE["name"]] == 0
            and event[data.MAP_SIGN_TYPE["name"]] == 0
            and event[data.MAP_COSMIC_CAT["name"]] == 0
            and event["t_sample_type"] == 0
        ):  # Beam nuel
            return event["w"]
        if (
            event[data.MAP_NU_TYPE["name"]] == 0
            and event[data.MAP_SIGN_TYPE["name"]] == 0
            and event[data.MAP_COSMIC_CAT["name"]] == 0
            and event["t_sample_type"] == 1
        ):  # Appeared nuel
            nu_energy = math.floor(event["t_nu_energy"] / 100)
            if nu_energy > 99:
                nu_energy = 99
            if nuel_osc[nu_energy] == 0.0:
                return event["w"]
            else:
                return nuel_osc[nu_energy] * event["w"]
        elif (
            event[data.MAP_NU_TYPE["name"]] == 0
            and event[data.MAP_SIGN_TYPE["name"]] == 1
            and event[data.MAP_COSMIC_CAT["name"]] == 0
            and event["t_sample_type"] == 0
        ):  # beam anuel
            return event["w"]
        elif (
            event[data.MAP_NU_TYPE["name"]] == 1
            and event[data.MAP_SIGN_TYPE["name"]] == 0
            and event[data.MAP_COSMIC_CAT["name"]] == 0
            and event["t_sample_type"] == 0
        ):  # Beam numu
            nu_energy = math.floor(event["t_nu_energy"] / 100)
            if nu_energy > 99:
                nu_energy = 99
            return numu_survival_prob[nu_energy] * event["w"]
        elif (
            event[data.MAP_NU_TYPE["name"]] == 1
            and event[data.MAP_SIGN_TYPE["name"]] == 1
            and event[data.MAP_COSMIC_CAT["name"]] == 0
            and event["t_sample_type"] == 0
        ):  # Beam anumu
            nu_energy = math.floor(event["t_nu_energy"] / 100)
            if nu_energy > 99:
                nu_energy = 99
            return numu_survival_prob[nu_energy] * event["w"]
        elif event[data.MAP_COSMIC_CAT["name"]] == 1 and event["t_sample_type"] == 2:
            return event["w"]
        else:
            return 0

    np.seterr(divide="ignore", invalid="ignore")
    tot_nuel = events[
        (events[data.MAP_NU_TYPE["name"]] == 0)
        & (events[data.MAP_SIGN_TYPE["name"]] == 0)
        & (events[data.MAP_COSMIC_CAT["name"]] == 0)
        & (events["t_sample_type"] == 0)
    ].shape[0]
    tot_anuel = events[
        (events[data.MAP_NU_TYPE["name"]] == 0)
        & (events[data.MAP_SIGN_TYPE["name"]] == 1)
        & (events[data.MAP_COSMIC_CAT["name"]] == 0)
        & (events["t_sample_type"] == 0)
    ].shape[0]
    tot_numu = events[
        (events[data.MAP_NU_TYPE["name"]] == 1)
        & (events[data.MAP_SIGN_TYPE["name"]] == 0)
        & (events[data.MAP_COSMIC_CAT["name"]] == 0)
        & (events["t_sample_type"] == 0)
    ].shape[0]
    tot_anumu = events[
        (events[data.MAP_NU_TYPE["name"]] == 1)
        & (events[data.MAP_SIGN_TYPE["name"]] == 1)
        & (events[data.MAP_COSMIC_CAT["name"]] == 0)
        & (events["t_sample_type"] == 0)
    ].shape[0]
    tot_cosmic = events[events[data.MAP_COSMIC_CAT["name"]] == 1].shape[0]

    if tot_nuel == 0:
        w_nuel = 0.0
    else:
        w_nuel = (1.0 / tot_nuel) * (nuel_frac * total_num)

    if tot_anuel == 0:
        w_anuel = 0.0
    else:
        w_anuel = (1.0 / tot_anuel) * (anuel_frac * total_num)

    if tot_numu == 0:
        w_numu = 0.0
    else:
        w_numu = (1.0 / tot_numu) * (numu_frac * total_num)

    if tot_anumu == 0:
        w_anumu = 0.0
    else:
        w_anumu = (1.0 / tot_anumu) * (anumu_frac * total_num)

    if tot_cosmic == 0:
        w_cosmic = 0.0
    else:
        w_cosmic = (1.0 / tot_cosmic) * (cosmic_frac * total_num)

    if verbose:
        print(
            "Weights: ({},{:.5f}), ({},{:.5f}), ({},{:.5f}), ({},{:.5f}), ({},{:.5f})".format(
                tot_nuel,
                w_nuel,
                tot_anuel,
                w_anuel,
                tot_numu,
                w_numu,
                tot_anumu,
                w_anumu,
                tot_cosmic,
                w_cosmic,
            )
        )

    events["w"] = events.apply(
        apply_scale_weight, axis=1, args=(w_nuel, w_anuel, w_numu, w_anumu, w_cosmic),
    )

    # Now we need to apply the oscillation probability weights
    osc_file = uproot.open(osc_file_name)

    # We need to scale the nuel events so they simulate the appearance spectra
    numu_ev = events[  # Get the unoscillated numu beam events
        (events[data.MAP_NU_TYPE["name"]] == 1)
        & (events[data.MAP_SIGN_TYPE["name"]] == 0)
        & (events[data.MAP_COSMIC_CAT["name"]] == 0)
        & (events["t_sample_type"] == 0)
    ]
    nuel_ev = events[  # Get the nuel events generated with the numu flux
        (events[data.MAP_NU_TYPE["name"]] == 0)
        & (events[data.MAP_SIGN_TYPE["name"]] == 0)
        & (events[data.MAP_COSMIC_CAT["name"]] == 0)
        & (events["t_sample_type"] == 1)
    ]

    numu_e_h = np.histogram(
        numu_ev["t_nu_energy"] / 100, bins=100, range=(0, 100), weights=numu_ev["w"],
    )
    nuel_e_h = np.histogram(
        nuel_ev["t_nu_energy"] / 100, bins=100, range=(0, 100), weights=nuel_ev["w"],
    )
    nuel_osc = (numu_e_h[0] * osc_file["hist_mue"].values) / nuel_e_h[0]

    # Apply a weight to every event
    events["w"] = events.apply(
        apply_osc_weight, axis=1, args=(osc_file["hist_mumu"].values, nuel_osc,),
    )

    return events


def apply_standard_cuts(
    events,
    cosmic_cut=0.001,
    q_cut=600.0,
    h_cut=600.0,
    theta_cut=0.65,
    phi_cut=0.25,
    verbose=False,
):
    """Calculate and apply the standard cuts to the events dataframe.

    Args:
        events (pd.DataFrame): events dataframe to append weights to
        cosmic_cut (float): cosmic classifier output cut
        q_cut (float): total collected event chargr cut
        h_cut (float): hough peak height cut
        theta_cut (float): reco theta direction cut
        phi_cut (float): reco phi direction cut
        verbose (bool): should we print the cut summary?

    Returns:
        pd.DataFrame: events with cuts applied
    """
    cosmic_cuts = np.zeros(len(events), dtype=bool)
    for column in events.columns:
        if "pred_t_cosmic_cat" in column:
            cosmic_cut_func = cut_apply(column, cosmic_cut, cut_type="greater")
            cosmic_cuts = events.apply(cosmic_cut_func, axis=1)
            events["cosmic_cut"] = cosmic_cuts
            print(events[events["t_cosmic_cat"] == 1][column].describe())

    q_cut_func = cut_apply("r_total_digi_q", q_cut, cut_type="lower")
    q_cuts = events.apply(q_cut_func, axis=1)

    h_cut_func = cut_apply("r_first_ring_height", h_cut, cut_type="lower")
    h_cuts = events.apply(h_cut_func, axis=1)

    theta_low_cut_func = cut_apply("r_dir_theta", -theta_cut, cut_type="lower")
    theta_low_cuts = events.apply(theta_low_cut_func, axis=1)

    theta_high_cut_func = cut_apply("r_dir_theta", theta_cut, cut_type="greater")
    theta_high_cuts = events.apply(theta_high_cut_func, axis=1)

    phi_low_cut_func = cut_apply("r_dir_phi", -phi_cut, cut_type="lower")
    phi_low_cuts = events.apply(phi_low_cut_func, axis=1)

    phi_high_cut_func = cut_apply("r_dir_phi", phi_cut, cut_type="greater")
    phi_high_cuts = events.apply(phi_high_cut_func, axis=1)

    events["cut"] = np.logical_or.reduce(
        (
            cosmic_cuts,
            q_cuts,
            h_cuts,
            theta_low_cuts,
            theta_high_cuts,
            phi_low_cuts,
            phi_high_cuts,
        )
    )

    if verbose:
        survived = []
        frac = []
        for i in range(len(data.MAP_FULL_COMB_CAT["labels"])):
            cat_events = events[events[data.MAP_FULL_COMB_CAT["name"]] == i]
            if len(cat_events) == 0:
                survived.append(0)
                frac.append(0)
            else:
                survived.append(len(cat_events[cat_events["cut"] == 0]))
                frac.append(survived[i] / len(cat_events))

        print(
            "Cuts:    ({},{:.5f}), ({},{:.5f}), ({},{:.5f}), ({},{:.5f})".format(
                survived[0],
                frac[0],
                survived[1],
                frac[1],
                survived[2],
                frac[2],
                survived[3],
                frac[3],
            )
        )

    return events


def cut_apply(variable, value, cut_type):
    """Create a function to apply a cut on a variable within the events dataframe.

    Args:
        variable (str): variable name to cut on
        value (float): cut value
        type (str): type (greater or lower)

    Returns:
        function: function to apply the cut to the dataframe
    """

    def cut_func(event):
        if cut_type == "greater":
            return event[variable] >= value
        elif cut_type == "lower":
            return event[variable] <= value
        else:
            raise NotImplementedError

    return cut_func


def calculate_curves(ev, thresholds=200, prefix=""):
    """Calculate efficiency, purity and figure of merit across the full range of cut values.

    Args:
        ev (pd.DataFrame): events dataframe to use
        cat_name (str): category name to use
        thresholds (int): number of threshold values to use
        prefix (str): prefix to apply to model output values

    Returns:
        output (dict): output dictionary of results
    """
    # selections = [
    #    ev[(ev["t_comb_cat"] == 0) & (ev["t_sample_type"] == 1)],  # Appeared CC nuel
    #    ev[(ev["t_comb_cat"] == 1)],  # All CC numu
    #    ev[(ev["t_comb_cat"] == 2)],  # NC
    # ]
    # keys = [
    #    prefix + "pred_t_comb_cat_0",
    #    prefix + "pred_t_comb_cat_1",
    #    prefix + "pred_t_comb_cat_2",
    # ]
    selections = [
        ev[(ev["t_comb_cat"] == 0)],  # CC nuel
        ev[(ev["t_comb_cat"] == 1)],  # CC numu
        ev[(ev["t_comb_cat"] == 2)],  # NC
    ]
    keys = [
        prefix + "pred_t_comb_cat_0",
        prefix + "pred_t_comb_cat_1",
        prefix + "pred_t_comb_cat_2",
    ]
    num_cats = len(selections)
    np.seterr(divide="ignore", invalid="ignore")

    cuts, totals = [], []
    max_foms_0, max_fom_cuts_0, max_fom_passed_0 = [], [], []
    max_foms_1, max_fom_cuts_1, max_fom_passed_1 = [], [], []
    foms_0, foms_1 = [], []
    sig_effs, bkg_effs, purities = [], [], []
    for cat in range(num_cats):
        totals.append(selections[cat]["w"].sum())
        max_foms_0.append(0.0)
        max_fom_cuts_0.append(0.0)
        max_foms_1.append(0.0)
        max_fom_cuts_1.append(0.0)
        max_fom_passed_0.append([])
        max_fom_passed_1.append([])
        sig_effs.append([])
        bkg_effs.append([])
        purities.append([])
        foms_0.append([])
        foms_1.append([])

    inc = float(1.0 / thresholds)
    for cut in range(thresholds + 1):
        cuts.append(cut * inc)
        for count_cat in range(num_cats):
            passed = []
            for cut_cat in range(num_cats):
                passed.append(
                    selections[cut_cat][
                        (selections[cut_cat]["cut"] == 0)
                        & (selections[cut_cat][keys[count_cat]] >= cuts[cut])
                    ]["w"].sum()
                )

            # Calculate the signal and background efficiencies for this category
            if passed[count_cat] == 0.0 or totals[count_cat] == 0.0:
                sig_effs[count_cat].append(0.0)
            else:
                sig_effs[count_cat].append(passed[count_cat] / totals[count_cat])
            bkg_passed = sum(passed[:count_cat] + passed[count_cat + 1 :])
            bkg_total = sum(totals[:count_cat] + totals[count_cat + 1 :])
            if bkg_passed == 0.0 or bkg_total == 0.0:
                bkg_effs[count_cat].append(0.0)
            else:
                bkg_effs[count_cat].append(bkg_passed / bkg_total)

            if passed[count_cat] == 0.0 or bkg_passed == 0.0:
                foms_1[count_cat].append(0.0)
            else:
                foms_1[count_cat].append(passed[count_cat] / math.sqrt(bkg_passed))

            if sig_effs[count_cat][cut] == 0.0 or bkg_effs[count_cat][cut] == 0.0:
                purities[count_cat].append(1.0)
                foms_0[count_cat].append(0.0)
            else:
                purities[count_cat].append(passed[count_cat] / sum(passed))
                foms_0[count_cat].append(
                    sig_effs[count_cat][cut] * purities[count_cat][cut]
                )

            if foms_0[count_cat][cut] > max_foms_0[count_cat]:
                max_foms_0[count_cat] = foms_0[count_cat][cut]
                max_fom_cuts_0[count_cat] = cuts[cut]
                max_fom_passed_0[count_cat] = [passed[cat] for cat in range(num_cats)]

            if foms_1[count_cat][cut] > max_foms_1[count_cat]:
                max_foms_1[count_cat] = foms_1[count_cat][cut]
                max_fom_cuts_1[count_cat] = cuts[cut]
                max_fom_passed_1[count_cat] = [passed[cat] for cat in range(num_cats)]

    # Convert the lists to numpy arrays
    cuts = np.asarray(cuts)
    sig_effs = np.asarray(sig_effs)
    bkg_effs = np.asarray(bkg_effs)
    purities = np.asarray(purities)
    foms_0 = np.asarray(foms_0)
    foms_1 = np.asarray(foms_1)

    # Calculate summary values
    sig_effs_auc = []
    bkg_effs_auc = []
    pur_auc = []
    fom_auc_0 = []
    fom_auc_1 = []
    roc_auc = []
    prc_auc = []
    for cat in range(num_cats):
        sig_effs_auc.append(auc(cuts, sig_effs[cat]))
        bkg_effs_auc.append(auc(cuts, bkg_effs[cat]))
        pur_auc.append(auc(cuts, purities[cat]))
        fom_auc_0.append(auc(cuts, foms_0[cat]))
        fom_auc_1.append(auc(cuts, foms_1[cat]))
        roc_auc.append(auc(bkg_effs[cat], sig_effs[cat]))
        prc_auc.append(auc(sig_effs[cat], purities[cat]))

    output = {
        "cuts": cuts,
        "sig_effs": sig_effs,
        "bkg_effs": bkg_effs,
        "purs": purities,
        "foms_0": foms_0,
        "foms_1": foms_1,
        "sig_effs_auc": sig_effs_auc,
        "bkg_effs_auc": bkg_effs_auc,
        "pur_auc": pur_auc,
        "fom_auc_0": fom_auc_0,
        "fom_auc_1": fom_auc_1,
        "roc_auc": roc_auc,
        "prc_auc": prc_auc,
        "max_foms_0": max_foms_0,
        "max_foms_1": max_foms_1,
        "max_fom_cuts_0": max_fom_cuts_0,
        "max_fom_cuts_1": max_fom_cuts_1,
        "max_fom_passed_0": max_fom_passed_0,
        "max_fom_passed_1": max_fom_passed_1,
    }

    return output


def calculate_eff_pur(ev, cut_value, e_bins=20, e_range=(0, 10000), prefix=""):
    """Calculate efficiency and purity curves at a specific cut value.

    Args:
        ev (pd.DataFrame): events dataframe to use
        cut_value (float): cut value to use
        e_bins (int): number of energy bins
        e_range: (tuple): energy range
        prefix (str): prefix to apply to model output values

    Returns:
        output (dict): output dictionary of results
    """
    """
    selections = [
        ev[
            (ev["t_comb_cat"] == 0)
            & (ev["t_sample_type"] == 1)
            & ((ev["t_all_cat"] == 0) | (ev["t_all_cat"] == 4))
        ],  # Appeared CC QEL nuel
        ev[
            (ev["t_comb_cat"] == 0)
            & (ev["t_sample_type"] == 1)
            & (
                (ev["t_all_cat"] == 1)
                | (ev["t_all_cat"] == 2)
                | (ev["t_all_cat"] == 3)
                | (ev["t_all_cat"] == 5)
            )
        ],  # Appeared CC nonQEL nuel
        ev[
            (ev["t_comb_cat"] == 1)
            & (ev["t_sample_type"] == 0)
            & ((ev["t_all_cat"] == 6) | (ev["t_all_cat"] == 10))
        ],  # Appeared CC QEL nuel
        ev[
            (ev["t_comb_cat"] == 1)
            & (ev["t_sample_type"] == 0)
            & (
                (ev["t_all_cat"] == 7)
                | (ev["t_all_cat"] == 8)
                | (ev["t_all_cat"] == 9)
                | (ev["t_all_cat"] == 11)
            )
        ],  # Appeared CC nonQEL nuel
        ev[(ev["t_comb_cat"] == 2)],  # NC
    ]
    keys = [
        prefix + "pred_t_comb_cat_0",
        prefix + "pred_t_comb_cat_0",
        prefix + "pred_t_comb_cat_1",
        prefix + "pred_t_comb_cat_1",
        prefix + "pred_t_comb_cat_2",
    ]
    cut_keys = [0, 0, 1, 1, 2]
    """
    selections = [
        ev[(ev["t_comb_cat"] == 0) & (ev["t_sample_type"] == 1)],  # Appeared CC nuel
        ev[(ev["t_comb_cat"] == 0) & (ev["t_sample_type"] == 0)],  # Beam CC nuel
        ev[(ev["t_comb_cat"] == 1) & (ev["t_sample_type"] == 0)],  # Survived CC numu
        ev[(ev["t_comb_cat"] == 2) & (ev["t_sample_type"] == 0)],  # Beam NC
    ]
    keys = [
        prefix + "pred_t_comb_cat_0",
        prefix + "pred_t_comb_cat_0",
        prefix + "pred_t_comb_cat_1",
        prefix + "pred_t_comb_cat_2",
    ]
    cut_keys = [0, 0, 1, 2]
    num_cats = len(selections)
    np.seterr(divide="ignore", invalid="ignore")

    fom_effs = []
    fom_purs = []
    for count_cat in range(num_cats):
        signal_h = None
        bkg_h = np.zeros(e_bins)
        bkg_err = np.zeros(e_bins)
        eff_hists = []
        for cut_cat in range(num_cats):
            total = selections[cut_cat]
            passed = selections[cut_cat][
                (selections[cut_cat]["cut"] == 0)
                & (
                    selections[cut_cat][keys[count_cat]]
                    > cut_value[cut_keys[count_cat]]
                )
            ]

            tot_h, tot_err, centers, edges = extended_hist(
                total["t_nu_energy"].to_numpy(),
                e_range[0],
                e_range[1],
                e_bins,
                weights=total["w"].to_numpy(),
            )

            pass_h, pass_err, centers, edges = extended_hist(
                passed["t_nu_energy"].to_numpy(),
                e_range[0],
                e_range[1],
                e_bins,
                weights=passed["w"].to_numpy(),
            )

            eff_h = np.divide(pass_h, tot_h)
            eff_err = np.multiply(
                np.sqrt(
                    np.add(
                        np.square(np.divide(pass_err, pass_h)),
                        np.square(np.divide(tot_err, tot_h)),
                    )
                ),
                eff_h,
            )
            eff_hists.append((eff_h, eff_err))

            if cut_cat == count_cat:
                signal_h = (pass_h, pass_err)
            else:
                bkg_h = np.add(bkg_h, pass_h)
                bkg_err = np.add(bkg_err, pass_err)

        fom_effs.append(eff_hists)

        pur_h = np.divide(
            signal_h[0],
            np.add(signal_h[0], bkg_h),
            out=np.zeros_like(bkg_h),
            where=(bkg_h != 0),
        )
        pur_err = np.multiply(
            np.sqrt(
                np.add(
                    np.square(np.divide(signal_h[1], signal_h[0])),
                    np.square(
                        np.divide(
                            np.add(bkg_err, signal_h[1]), np.add(bkg_h, signal_h[0])
                        )
                    ),
                )
            ),
            pur_h,
        )

        fom_purs.append((pur_h, pur_err))

    output = {
        "fom_effs": fom_effs,
        "fom_purs": fom_purs,
    }

    return output


def classify(event, categories, prefix):
    """Classify the event by the highest score.

    Args:
        event (dict): pandas event(row) dict
        categories (int): number of categories
        prefix: (str): prefix of the different scores for this classification

    Returns:
        int: category classification
    """
    if categories == 1:
        return round(event[prefix])
    elif "cosmic_cut" in event and event["cosmic_cut"]:
        return categories
    else:
        x = [event[prefix + str(i)] for i in range(categories)]
        return np.asarray(x).argmax()


def run_pca(
    events,
    model,
    seperate_channels=True,
    reco_pars=True,
    layer_name="dense_final",
    standardise=True,
    components=3,
    max_events=10000,
    verbose=False,
):
    """Run PCA on the final dense layer outputs and append results to events dataframe.

    Args:
        events (pd.DataFrame): events dataframe to append outputs
        model (chipsnet.Model): model to use for prediction
        layer_name (str): final dense layer name
        standardise (bool): should we apply a standard scalar to the dense layer outputs?
        components (int): number of PCA componenets to calculate
        max_events (int): maximum number of events to use in PCA
        verbose (bool): should we print the PCA summary?

    Returns:
        pd.DataFrame: events dataframe with PCA outputs
    """
    # Create model that outputs the dense layer outputs
    dense_model = Model(
        inputs=model.model.input, outputs=model.model.get_layer(layer_name).output
    )

    inputs = [np.stack(events["image_0"].to_numpy())]
    if seperate_channels:
        try:
            inputs.append(np.stack(events["image_1"].to_numpy()))
        except Exception:
            pass
        try:
            inputs.append(np.stack(events["image_2"].to_numpy()))
        except Exception:
            pass

    if reco_pars:
        inputs.append(np.stack(events["r_vtx_x"].to_numpy()))
        inputs.append(np.stack(events["r_vtx_y"].to_numpy()))
        inputs.append(np.stack(events["r_vtx_z"].to_numpy()))
        inputs.append(np.stack(events["r_dir_theta"].to_numpy()))
        inputs.append(np.stack(events["r_dir_phi"].to_numpy()))

    # Make the predictions
    dense_outputs = dense_model.predict(inputs)

    # Run a standard scalar on the dense outputs if needed
    if standardise:
        dense_outputs = StandardScaler().fit_transform(dense_outputs)

    # Run the PCA
    pca = PCA(n_components=components)
    pca_result = pca.fit_transform(dense_outputs[:max_events])
    pca_result = pd.DataFrame(pca_result)

    # Append the results to the events dataframe
    for component in range(components):
        events["pca" + str(component)] = pca_result[component]

    if verbose:
        print(
            "Explained variation per principal component: {}".format(
                pca.explained_variance_ratio_
            )
        )

    return events


def run_tsne(
    events,
    model,
    seperate_channels=True,
    reco_pars=True,
    layer_name="dense_final",
    standardise=True,
    components=3,
    max_events=10000,
):
    """Run t-SNE on the final dense layer outputs and append results to events dataframe.

    Args:
        events (pd.DataFrame): events dataframe to append outputs
        model (chipsnet.Model): model to use for prediction
        layer_name (str): final dense layer name
        standardise (bool): should we apply a standard scalar to the dense layer outputs?
        components (int): number of t-SNE componenets to calculate
        max_events (int): maximum number of events to use in t-SNE

    Returns:
        pd.DataFrame: events dataframe with t-SNE outputs
    """
    # Create model that outputs the dense layer outputs
    dense_model = Model(
        inputs=model.model.input, outputs=model.model.get_layer(layer_name).output
    )

    inputs = [np.stack(events["image_0"].to_numpy())]
    if seperate_channels:
        try:
            inputs.append(np.stack(events["image_1"].to_numpy()))
        except Exception:
            pass

        try:
            inputs.append(np.stack(events["image_2"].to_numpy()))
        except Exception:
            pass
    if reco_pars:
        inputs.append(np.stack(events["r_vtx_x"].to_numpy()))
        inputs.append(np.stack(events["r_vtx_y"].to_numpy()))
        inputs.append(np.stack(events["r_vtx_z"].to_numpy()))
        inputs.append(np.stack(events["r_dir_theta"].to_numpy()))
        inputs.append(np.stack(events["r_dir_phi"].to_numpy()))

    # Make the predictions
    dense_outputs = dense_model.predict(inputs)

    # Run a standard scalar on the dense outputs if needed
    if standardise:
        dense_outputs = StandardScaler().fit_transform(dense_outputs)

    # Run t-SNE
    tsne = TSNE(n_components=components, verbose=1, perplexity=40, n_iter=300)
    tsne_result = tsne.fit_transform(dense_outputs[:max_events])
    tsne_result = pd.DataFrame(tsne_result)

    # Append the results to the events dataframe
    for component in range(components):
        events["tsne" + str(component)] = tsne_result[component]

    return events


def explain_gradcam(
    events, model, num_events, output="t_all_cat", layer_name="path0_block1"
):
    """Run GradCAM on the given model using the events.

    Args:
        events (pd.DataFrame): events dataframe to append outputs
        model (chipsnet.Model): model to use for prediction
        num_events (int): number of events to run
        output (str): single classification model output to use
        layer_name (str): model layer name to use

    Returns:
        list[event_outputs]: list of GradCAM outputs
    """
    explain_m = Model(
        inputs=model.model.input, outputs=model.model.get_layer(output).output
    )
    outputs = []
    for event in range(num_events):
        category = int(events[output][event])
        image = tf.expand_dims(events["image_0"][event], axis=0).numpy()
        outputs.append(
            GradCAM().explain(
                (image, category),
                explain_m,
                class_index=category,
                layer_name=layer_name,
            )
        )
    return outputs


def explain_occlusion(events, model, num_events, output="t_all_cat"):
    """Run OcclusionSensitivity on the given model using the events.

    Args:
        events (pd.DataFrame): events dataframe to append outputs
        model (chipsnet.Model): model to use for prediction
        num_events (int): number of events to run
        output (str): single classification model output to use

    Returns:
        list[event_outputs]: list of OcclusionSensitivity outputs
    """
    explain_m = Model(
        inputs=model.model.input, outputs=model.model.get_layer(output).output
    )
    outputs = []
    for event in range(num_events):
        category = int(events[output][event])
        image = tf.expand_dims(events["image_0"][event], axis=0).numpy()
        outputs.append(
            OcclusionSensitivity().explain(
                (image, category), explain_m, class_index=category, patch_size=3
            )
        )
    return outputs


def explain_activation(
    events, model, num_events, output="t_all_cat", layer_name="path0_block1_conv0"
):
    """Run ExtractActivations on the given model using the events.

    Args:
        events (pd.DataFrame): events dataframe to append outputs
        model (chipsnet.Model): model to use for prediction
        num_events (int): number of events to run
        output (str): single classification model output to use
        layer_name (str): model layer name to use

    Returns:
        list[event_outputs]: list of ExtractActivations outputs
    """
    explain_m = Model(
        inputs=model.model.input, outputs=model.model.get_layer(output).output
    )
    outputs = []
    for event in range(num_events):
        category = int(events[output][event])
        image = tf.expand_dims(events["image_0"][event], axis=0).numpy()
        outputs.append(
            ExtractActivations().explain(
                (image, category), explain_m, layers_name=layer_name
            )
        )
    return outputs


def explain_grads(events, model, num_events, output="t_all_cat"):
    """Run various gradient explain methods on the given model using the events.

    Args:
        events (pd.DataFrame): events dataframe to append outputs
        model (chipsnet.Model): model to use for prediction
        num_events (int): number of events to run
        output (str): single classification model output to use

    Returns:
        dict: Dictionary of gradient outputs
    """
    explain_m = Model(
        inputs=model.model.input, outputs=model.model.get_layer(output).output
    )
    outputs = {"vanilla": [], "smooth": [], "integrated": [], "inputs": []}
    for event in range(num_events):
        category = int(events[output][event])
        image = tf.expand_dims(events["image_0"][event], axis=0).numpy()
        outputs["vanilla"].append(
            VanillaGradients().explain(
                (image, category), explain_m, class_index=category
            )
        )
        outputs["smooth"].append(
            SmoothGrad().explain((image, category), explain_m, class_index=category)
        )
        outputs["integrated"].append(
            IntegratedGradients().explain(
                (image, category), explain_m, class_index=category
            )
        )
        outputs["inputs"].append(
            GradientsInputs().explain(
                (image, category), explain_m, class_index=category
            )
        )
    return outputs


def print_globes_effs(outputs):
    """Print the GLoBES efficiency arrays.

    Args:
        outputs (dict): outputs to use
    """
    print("---- nuel-cc selection")
    print(["{0:0.4f}".format(i) for i in outputs["fom_effs"][0][0][0]])
    print(["{0:0.4f}".format(i) for i in outputs["fom_effs"][0][1][0]])
    print(["{0:0.4f}".format(i) for i in outputs["fom_effs"][0][2][0]])
    print(["{0:0.4f}".format(i) for i in outputs["fom_effs"][0][3][0]])
    print(["{0:0.4f}".format(i) for i in outputs["fom_effs"][0][4][0]])
    print("---- numu-cc selection")
    print(["{0:0.4f}".format(i) for i in outputs["fom_effs"][2][0][0]])
    print(["{0:0.4f}".format(i) for i in outputs["fom_effs"][2][1][0]])
    print(["{0:0.4f}".format(i) for i in outputs["fom_effs"][2][2][0]])
    print(["{0:0.4f}".format(i) for i in outputs["fom_effs"][2][3][0]])
    print(["{0:0.4f}".format(i) for i in outputs["fom_effs"][2][4][0]])
    print("---- nc selection")
    print(["{0:0.4f}".format(i) for i in outputs["fom_effs"][4][0][0]])
    print(["{0:0.4f}".format(i) for i in outputs["fom_effs"][4][1][0]])
    print(["{0:0.4f}".format(i) for i in outputs["fom_effs"][4][2][0]])
    print(["{0:0.4f}".format(i) for i in outputs["fom_effs"][4][3][0]])
    print(["{0:0.4f}".format(i) for i in outputs["fom_effs"][4][4][0]])


def print_globes_smearing(ev, hists, names):
    """Output Globes compatible energy smearing files.

    Args:
        hists (list[numpy 2darray]): list of histogram energy smearing arrays
        names (list[str]): event type name
    """
    for hist, name in zip(hists, names):
        with open("smearing/" + name + ".dat", "w", encoding="utf-8") as f:
            f.write("energy(#" + name + ")<\n")
            f.write("@energy =\n")
            for i in range(40):
                f.write("{0, 40, ")
                for j in range(40):
                    if j == 39:
                        f.write(str.format("{0:.5f}", hist[j, i]))
                    else:
                        f.write(str.format("{0:.5f}, ", hist[j, i]))
                f.write("}:\n")
            f.write(">\n")


def extended_hist(
    data, xmin, xmax, nbins, underflow=False, overflow=False, weights=None
):
    """Create a histogram and associated errors in numpy.

    Args:
        data (np.array): numpy array of input data
        xmin (float): minimum value
        xmax (float): maximum value
        nbins (int): number of bins
        underflow (bool): should we include an underflow bin?
        overflow (bool): should we include an overflow bin?
        weights (np.array): associated weight values for the data entries

    Returns:
        np.array: bin values
        np.array: error values for each bin
        np.array: bin centers
        np.array: bin edges
    """
    if weights is not None:
        if weights.shape != data.shape:
            raise ValueError(
                "Unequal shapes data: {}; weights: {}".format(data.shape, weights.shape)
            )
    edges = np.linspace(xmin, xmax, nbins + 1)
    neginf = np.array([-np.inf], dtype=np.float32)
    posinf = np.array([np.inf], dtype=np.float32)
    uselsp = np.concatenate([neginf, edges, posinf])
    if weights is None:
        hist, bin_edges = np.histogram(data, bins=uselsp)
    else:
        hist, bin_edges = np.histogram(data, bins=uselsp, weights=weights)

    n = hist[1:-1]
    if underflow:
        n[0] += hist[0]
    if overflow:
        n[-1] += hist[-1]

    if weights is None:
        w = np.sqrt(n)
    else:
        bin_sumw2 = np.zeros(nbins + 2, dtype=np.float32)
        digits = np.digitize(data, edges)
        for i in range(nbins + 2):
            bin_sumw2[i] = np.sum(np.power(weights[np.where(digits == i)[0]], 2))
        w = bin_sumw2[1:-1]
        if underflow:
            w[0] += bin_sumw2[0]
        if overflow:
            w[-1] += bin_sumw2[-1]
        w = np.sqrt(w)

    centers = np.delete(edges, [0]) - (np.ediff1d(edges) / 2.0)
    return n, w, centers, edges


def gaussian(x, a, b, c):
    """Gaussian function to fit histograms with.

    Args:
        x (float): input value
        a (str): scaling factor
        b (int): exponent
        c (int): denomenator

    Returns:
        float: result of gaussian function
    """
    val = a * np.exp(-((x - b) ** 2) / c ** 2)
    return val


def frac_e_vs_par(
    events,
    par="t_nu_energy",
    low=1000,
    high=8000,
    bin_size=1000,
    fit_name="frac_nu_energy",
):
    """Bin fractional energy resolution in by a parameter.

    Args:
        events (pd.Dataframe): events dataframe
        par (str): parameter
        low (int): low range
        high (int): high range
        bin_size (int): size of bins
        fit_name (str): frac energy to fit

    Returns:
        (list, list): list of cuts, list of std errors
    """
    cuts, mean_list, mean_err_list, std_list, std_err_list = [], [], [], [], []
    for cut in range(low, high, bin_size):
        try:
            subset = events[(events[par] >= cut)]
            subset = subset[(subset[par] <= (cut + bin_size))]
            n, bins = np.histogram(
                subset[fit_name],
                bins=40,
                range=(-1, 1),
                weights=subset["w"],
                density=True,
            )
            centers = 0.5 * (bins[1:] + bins[:-1])
            pars, cov = curve_fit(
                gaussian,
                centers,
                n,
                p0=[1, subset[fit_name].mean(), subset[fit_name].std()],
            )
            mean_list.append(abs(pars[1]))
            mean_err_list.append(abs(np.sqrt(cov[1, 1])))
            std_list.append(abs(pars[2]))
            std_err_list.append(abs(np.sqrt(cov[2, 2])))
            cuts.append(cut + (bin_size / 2))
        except Exception as e:
            print(e)
            cuts.append(cut + (bin_size / 2))
            mean_list.append(0)
            mean_err_list.append(0)
            std_list.append(0)
            std_err_list.append(0)
    return np.array(cuts), [mean_list, mean_err_list], [std_list, std_err_list]


def get_old_df(file_name, l_type):
    """Generate DataFrame from old reco/pid files.

    Args:
        file_name (str): file names
        l_type (int): lepton type (11 or 13)

    Returns:
        pd.DataFrame: dataframe with similar fields to standard events df
    """
    f = uproot.open(file_name)
    true = f["fResultsTree"]["TruthInfo"]
    reco = f["fResultsTree"]["PidInfo_ElectronLike"]
    if l_type == 13:
        reco = f["fResultsTree"]["PidInfo_MuonLike"]
    pid = f["PIDTree_ann"]

    df = pd.DataFrame()
    df["t_nu_energy"] = true["fBeamEnergy"].array()
    df["t_vtx_x"] = true["fVtxX"].array() / 1000
    df["t_vtx_y"] = true["fVtxY"].array() / 1000
    df["t_vtx_z"] = true["fVtxZ"].array() / 1000
    df["t_vtx_t"] = true["fVtxTime"].array()
    df["r_vtx_x"] = reco["fVtxX"].array() / 100
    df["r_vtx_y"] = reco["fVtxY"].array() / 100
    df["r_vtx_z"] = reco["fVtxZ"].array() / 100
    df["r_vtx_t"] = reco["fVtxTime"].array()
    df["is_cc"] = true["fIsCC"].array()
    df["is_qe"] = true["fIsQE"].array()
    df["r_lep_energy"] = reco["fEnergy"].array()

    # Find the true lepton energies
    t_lep_energy = []
    for cc, energy, pdg in zip(
        true["fIsCC"].array(),
        true["fPrimaryEnergies"].array(),
        true["fPrimaryPDGs"].array(),
    ):
        if cc:
            try:
                t_lep_energy.append(energy[np.where(pdg == l_type)][0])
            except Exception:
                t_lep_energy.append(0.0)
        else:
            t_lep_energy.append(0.0)
    df["t_lep_energy"] = np.asarray(t_lep_energy)

    # Merge the output PID info with the dataframe
    pid_df = pd.DataFrame()
    pid_df["t_nu_energy"] = pid["trueBeamE"].array()
    pid_df["ann_vs_numu"] = pid["annNueCCQEvsNumuCCQE"].array()
    pid_df["ann_vs_nc"] = pid["annNueCCQEvsNC"].array()
    pid_df["delta_charge_2lnl"] = pid["deltaCharge2LnL"].array()
    pid_df["delta_time_2lnl"] = pid["deltaTime2LnL"].array()
    df = pd.merge(df, pid_df, how="inner", on=["t_nu_energy"])

    # Calculate some things
    df["frac_lep_e"] = (df["r_lep_energy"] - df["t_nu_energy"]) / df["t_nu_energy"]
    df["l_type"] = np.full(len(df), l_type)

    # Apply the correct weights to each event
    if l_type == 11:
        df["t_nu_type"] = np.full(len(df), 0)
        df["t_sign_type"] = np.full(len(df), 0)
        df["t_cosmic_cat"] = np.full(len(df), 0)
        df["t_sample_type"] = np.full(len(df), 1)
    elif l_type == 13:
        df["t_nu_type"] = np.full(len(df), 1)
        df["t_sign_type"] = np.full(len(df), 0)
        df["t_cosmic_cat"] = np.full(len(df), 0)
        df["t_sample_type"] = np.full(len(df), 0)

    return df
