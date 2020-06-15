# -*- coding: utf-8 -*-

"""Utility module containing lots of helpful methods for evaluation and plotting
"""

import os
import time
import copy
import ast

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tqdm import tqdm
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

import chipsnet.config
import chipsnet.data as data
import chipsnet.models


def data_from_conf(config, name):
    """Get the named data reader as defined in the configuration.
    Args:
        config (dotmap.DotMap): Base configuration namespace
        name (str): Data name
    Returns:
        chipsnet.data reader: Data reader from configuration
    """
    # We copy the configuration so we don't modify it
    data_config = copy.copy(config)
    data_config.data = config.samples[name]
    data = chipsnet.data.Reader(data_config)
    return data


def model_from_conf(config, name):
    """Get the loaded model corresponding to the given name and configuration.
    Args:
        config (dotmap.DotMap): Base configuration namespace
        name (str): Model name
    Returns:
        chipsnet.models model: Model from the configuration
    """
    # We copy the configuration so we don't modify it
    model_config = copy.copy(config)
    model_config.model = config.models[name]
    model_config.exp.output_dir = config.models[name].dir
    model_config.exp.name = config.models[name].path
    model_config.data.channels = config.models[name].channels
    chipsnet.config.setup_dirs(model_config, False)
    model = chipsnet.models.get_model(model_config)
    model.load()
    return model


def model_history(config, name):
    """Get the model training history.
    Args:
        config (dotmap.DotMap): Base configuration namespace
        name (str): Model name
    Returns:
        pandas.DataFrame: Dataframe containing model training history
    """
    model_config = copy.copy(config)
    model_config.model = config.models[name]
    model_config.exp.output_dir = config.models[name].dir
    model_config.exp.name = config.models[name].path
    model_config.data.channels = config.models[name].channels
    chipsnet.config.setup_dirs(model_config, False)
    history = pd.read_csv(os.path.join(model_config.exp.exp_dir, "history.csv"))
    history_dict = {}
    for key in history.keys():
        history_dict[key] = ast.literal_eval(history[key].values[0])
    return pd.DataFrame.from_dict(history_dict)


def process_ds(config, data_name, model_names, verbose=False):
    """Fully process a dataset through a list of models and run standard evaluation.
    Args:
        config (dotmap.DotMap): Base configuration namespace
        data_name (str): Data name
        model_names (List[str]): Model names
        verbose (bool): Should processing print summaries?
    Returns:
        pandas.DataFrame: Events dataframe fully processed
    """
    print("Processing {}... ".format(data_name), end="", flush=True)
    start_time = time.time()

    # Get the dataframe from the dataset name
    events = data_from_conf(config, data_name).testing_df(config.eval.examples)

    # Run all the required inference
    for model_name in model_names:
        model = model_from_conf(config, model_name)
        events = run_inference(events, model, prefix=model_name + "_")

    # Apply the event weights
    events = apply_weights(
        events,
        total_num=config.eval.weights.total,
        nuel_frac=config.eval.weights.nuel,
        anuel_frac=config.eval.weights.anuel,
        numu_frac=config.eval.weights.numu,
        anumu_frac=config.eval.weights.anumu,
        cosmic_frac=config.eval.weights.cosmic,
        verbose=verbose,
    )

    # Apply the standard cuts
    events = apply_standard_cuts(
        events,
        cosmic_cut=config.eval.cuts.cosmic,
        q_cut=config.eval.cuts.q,
        h_cut=config.eval.cuts.h,
        theta_cut=config.eval.cuts.theta,
        phi_cut=config.eval.cuts.phi,
        verbose=verbose,
    )

    # Classify into fully combined categories and print the classification reports
    outputs = {
        "cuts": [],
        "sig_effs": [],
        "bkg_effs": [],
        "purs": [],
        "foms": [],
        "fom_effs": [],
        "fom_purs": [],
        "sig_effs_auc": [],
        "bkg_effs_auc": [],
        "pur_auc": [],
        "fom_auc": [],
        "roc_auc": [],
        "comb_matrices": [],
        "all_matrices": [],
        "report": [],
    }
    for model_name in model_names:
        if "cosmic" in model_name or "cos" in model_name:
            continue

        # Combine categories into fully combined ones
        events = full_comb_combine(events, prefix=model_name + "_")

        # Run classification and print report
        class_prefix = model_name + "_" + "pred_t_comb_cat_"
        events[model_name + "_comb_cat_class"] = events.apply(
            classify, axis=1, args=(3, class_prefix)
        )
        class_prefix = model_name + "_" + "pred_t_all_cat_"
        events[model_name + "_all_cat_class"] = events.apply(
            classify, axis=1, args=(24, class_prefix)
        )

        report = classification_report(
            events["t_comb_cat"],
            events[model_name + "_comb_cat_class"],
            target_names=["nuel-cc", "numu-cc", "nc"],
        )
        outputs["report"].append(report)
        if verbose:
            print(report)

        # Run curve calculation
        curves_output = calculate_curves(
            events, prefix=model_name + "_", verbose=verbose
        )
        outputs["cuts"].append(curves_output["cuts"])
        outputs["sig_effs"].append(curves_output["sig_effs"])
        outputs["bkg_effs"].append(curves_output["bkg_effs"])
        outputs["purs"].append(curves_output["purs"])
        outputs["foms"].append(curves_output["foms"])
        outputs["fom_effs"].append(curves_output["fom_effs"])
        outputs["fom_purs"].append(curves_output["fom_purs"])
        outputs["sig_effs_auc"].append(curves_output["sig_effs_auc"])
        outputs["bkg_effs_auc"].append(curves_output["bkg_effs_auc"])
        outputs["pur_auc"].append(curves_output["pur_auc"])
        outputs["fom_auc"].append(curves_output["fom_auc"])
        outputs["roc_auc"].append(curves_output["roc_auc"])

        matrix_comb = confusion_matrix(
            events["t_comb_cat"],
            events[model_name + "_comb_cat_class"],
            normalize="true",
        )
        matrix_comb = np.rot90(matrix_comb, 1)
        matrix_comb = pd.DataFrame(matrix_comb)
        outputs["comb_matrices"].append(matrix_comb)
        matrix_all = confusion_matrix(
            events["t_all_cat"], events[model_name + "_all_cat_class"], normalize="true"
        )
        matrix_all = np.rot90(matrix_all, 1)
        matrix_all = pd.DataFrame(matrix_all)
        outputs["all_matrices"].append(matrix_all)

    print("took {:.2f} seconds".format(time.time() - start_time))

    # Return everything
    return events, outputs


def run_inference(events, model, prefix=""):
    """Run predictions on the input dataset and append outputs to events dataframe.
    Args:
        events (pandas.DataFrame): Events dataframe to append outputs
        model (chipsnet.Model): Model to use for prediction
        prefix (str): Prefix to append to column name
    Returns:
        pandas.DataFrame: Events dataframe with model predictions
    """
    # Make the predictions
    outputs = model.model.predict(np.stack(events["image_0"].to_numpy()))

    # Append the predictions to the events dataframe
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
            events[base_key] = output
        else:
            for cat in range(output.shape[1]):
                key = base_key + "_" + str(cat)
                events[key] = output[:, cat]

    return events


def full_comb_combine(events, prefix=""):
    """Combine t_all_cat scores into t_comb_cat scores.
    Args:
        events (pandas.DataFrame): Events dataframe to calculate combined category scores
        prefix (str): Prefix to apply to model output values
    Returns:
        pandas.DataFrame: Events dataframe with combined category scores
    """

    def full_comb_apply(event, apply_prefix):
        nuel_cc_cats = [0, 1, 2, 3, 4, 5]
        nuel_cc_value = 0.0
        for cat in nuel_cc_cats:
            nuel_cc_value = nuel_cc_value + event[str(apply_prefix) + str(cat)]

        numu_cc_cats = [12, 13, 14, 15, 16, 17]
        numu_cc_value = 0.0
        for cat in numu_cc_cats:
            numu_cc_value = numu_cc_value + event[apply_prefix + str(cat)]

        nc_cats = [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]
        nc_value = 0.0
        for cat in nc_cats:
            nc_value = nc_value + event[apply_prefix + str(cat)]

        return nuel_cc_value, numu_cc_value, nc_value

    apply_prefix = prefix + "pred_t_all_cat_"
    comb_prefix = prefix + "pred_t_comb_cat_"
    events["scores"] = events.apply(full_comb_apply, axis=1, args=(apply_prefix,))
    events[comb_prefix + "0"] = events.scores.map(lambda x: x[0])
    events[comb_prefix + "1"] = events.scores.map(lambda x: x[1])
    events[comb_prefix + "2"] = events.scores.map(lambda x: x[2])
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
    verbose=False,
):
    """Calculate and apply the 'weight' column to scale events to predicted numbers.
    Args:
        events (pandas.DataFrame): Events dataframe to append weights to
        total_num (float): Total number of expected events in a year
        nuel_frac (float): Fraction of events from nuel
        anuel_frac (float): Fraction of events from anuel
        numu_frac (float): Fraction of events from numu
        anumu_frac (float): Fraction of events from anumu
        cosmic_frac (float): Fractions of events from cosmics
        verbose (bool): Should we print weight summary?
    Returns:
        pandas.DataFrame: Events dataframe with weights
    """

    def add_weight(event, w_nuel, w_anuel, w_numu, w_anumu, w_cosmic):
        """Add the correct weight to each event.
        Args:
            event (dict): Pandas event(row) dict
            w_nuel: nuel weight
            w_anuel: anuel weight
            w_numu: numu weight
            w_anumu: anumu weight
            w_cosmic: cosmic weight
        """
        if (
            event[data.MAP_NU_TYPE.name] == 0
            and event[data.MAP_SIGN_TYPE.name] == 0
            and event[data.MAP_COSMIC_CAT.name] == 0
        ):
            return w_nuel
        elif (
            event[data.MAP_NU_TYPE.name] == 0
            and event[data.MAP_SIGN_TYPE.name] == 1
            and event[data.MAP_COSMIC_CAT.name] == 0
        ):
            return w_anuel
        elif (
            event[data.MAP_NU_TYPE.name] == 1
            and event[data.MAP_SIGN_TYPE.name] == 0
            and event[data.MAP_COSMIC_CAT.name] == 0
        ):
            return w_numu
        elif (
            event[data.MAP_NU_TYPE.name] == 1
            and event[data.MAP_SIGN_TYPE.name] == 1
            and event[data.MAP_COSMIC_CAT.name] == 0
        ):
            return w_anumu
        elif event[data.MAP_COSMIC_CAT.name] == 1:
            return w_cosmic
        else:
            raise NotImplementedError

    tot_nuel = events[
        (events[data.MAP_NU_TYPE.name] == 0)
        & (events[data.MAP_SIGN_TYPE.name] == 0)
        & (events[data.MAP_COSMIC_CAT.name] == 0)
    ].shape[0]
    tot_anuel = events[
        (events[data.MAP_NU_TYPE.name] == 0)
        & (events[data.MAP_SIGN_TYPE.name] == 1)
        & (events[data.MAP_COSMIC_CAT.name] == 0)
    ].shape[0]
    tot_numu = events[
        (events[data.MAP_NU_TYPE.name] == 1)
        & (events[data.MAP_SIGN_TYPE.name] == 0)
        & (events[data.MAP_COSMIC_CAT.name] == 0)
    ].shape[0]
    tot_anumu = events[
        (events[data.MAP_NU_TYPE.name] == 1)
        & (events[data.MAP_SIGN_TYPE.name] == 1)
        & (events[data.MAP_COSMIC_CAT.name] == 0)
    ].shape[0]
    tot_cosmic = events[events[data.MAP_COSMIC_CAT.name] == 1].shape[0]

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

    events["w"] = events.apply(
        add_weight, axis=1, args=(w_nuel, w_anuel, w_numu, w_anumu, w_cosmic)
    )

    if verbose:
        print(
            "Nuel:   {}, weight: {:.5f}, actual: {:.2f}".format(
                tot_nuel, w_nuel, total_num * nuel_frac
            )
        )
        print(
            "Anuel:  {}, weight: {:.5f}, actual: {:.2f}".format(
                tot_anuel, w_anuel, total_num * anuel_frac
            )
        )
        print(
            "Numu:   {}, weight: {:.5f}, actual: {:.2f}".format(
                tot_numu, w_numu, total_num * numu_frac
            )
        )
        print(
            "Anumu:  {}, weight: {:.5f}, actual: {:.2f}".format(
                tot_anumu, w_anumu, total_num * anumu_frac
            )
        )
        print(
            "Cosmic: {}, weight: {:.3f}, actual: {:.2f}".format(
                tot_cosmic, w_cosmic, total_num * cosmic_frac
            )
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
        events (pandas.DataFrame): Events dataframe to append weights to
        cosmic_cut (float): Cosmic classifier output cut
        q_cut (float): Total collected event chargr cut
        h_cut (float): Hough peak height cut
        theta_cut (float): Reco theta direction cut
        phi_cut (float): Reco phi direction cut
        verbose (bool): Should we print cut summary?
    Returns:
        events (pandas.DataFrame): Events with cuts applied
    """
    cosmic_cuts = np.zeros(len(events), dtype=bool)
    if "pred_t_cosmic_cat" in events.columns:
        cosmic_cut_func = cut_apply("pred_t_cosmic_cat", cosmic_cut, type="greater")
        cosmic_cuts = events.apply(cosmic_cut_func, axis=1)

    q_cut_func = cut_apply("r_raw_total_digi_q", q_cut, type="lower")
    q_cuts = events.apply(q_cut_func, axis=1)

    h_cut_func = cut_apply("r_first_ring_height", h_cut, type="lower")
    h_cuts = events.apply(h_cut_func, axis=1)

    theta_low_cut_func = cut_apply("r_dirTheta", -theta_cut, type="lower")
    theta_low_cuts = events.apply(theta_low_cut_func, axis=1)

    theta_high_cut_func = cut_apply("r_dirTheta", theta_cut, type="greater")
    theta_high_cuts = events.apply(theta_high_cut_func, axis=1)

    phi_low_cut_func = cut_apply("r_dirPhi", -phi_cut, type="lower")
    phi_low_cuts = events.apply(phi_low_cut_func, axis=1)

    phi_high_cut_func = cut_apply("r_dirPhi", phi_cut, type="greater")
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
        for i in range(len(data.MAP_FULL_COMB_CAT.labels)):
            cat_events = events[events[data.MAP_FULL_COMB_CAT.name] == i]
            survived_events = cat_events[cat_events["cut"] == 0]
            if cat_events.shape[0] == 0:
                continue
            print(
                "{}: total {}, survived: {}".format(
                    data.MAP_FULL_COMB_CAT.labels[i],
                    cat_events.shape[0],
                    survived_events.shape[0] / cat_events.shape[0],
                )
            )

    return events


def cut_apply(variable, value, type):
    """Creates a function to apply a cut on a variable within the events dataframe
    Args:
        variable (str): Variable name to cut on
        value (float): Cut value
        type (str): Type (greater or lower)
    Returns:
        function: Function to apply the cut to the dataframe
    """

    def cut_func(event):
        if type == "greater":
            return event[variable] >= value
        elif type == "lower":
            return event[variable] <= value
        else:
            raise NotImplementedError

    return cut_func


def calculate_curves(
    events, cat_name="t_comb_cat", thresholds=200, prefix="", verbose=False
):
    """Calculate efficiency, purity and figure of merit across the full range of cut values
    Args:
        events (pandas.DataFrame): Events dataframe to use
        cat_name (str): Category name to use
        thresholds (int): Number of threshold values to use
        prefix (str): Prefix to apply to model output values
        verbose (bool): Should we print summary?
    Returns:
        cuts (np.array): Array of threshold cut values
        sig_effs (np.array): Array signal efficiencies
        bkg_effs (np.array): Array background efficiencies
        purities (np.array): Array of purities
        foms (np.array): Array of figure-of-merits (efficiency*purity)
    """
    num_cats = chipsnet.data.get_map(cat_name)["categories"]
    prefix = prefix + "pred_" + cat_name + "_"

    cuts, totals, max_foms, max_fom_cuts = [], [], [], []
    sig_effs, bkg_effs, purities, foms = [], [], [], []
    for cat in range(num_cats):
        totals.append(events[events[cat_name] == cat]["w"].sum())
        max_foms.append(0.0)
        max_fom_cuts.append(0.0)

    for cat in range(num_cats):
        sig_effs.append([1.0])
        bkg_effs.append([1.0])
        purities.append([totals[cat] / sum(totals)])
        foms.append([sig_effs[cat][0] * purities[cat][0]])

    cuts.append(0.0)
    inc = float(1.0 / thresholds)
    for cut in range(thresholds):
        cuts.append((cut * inc) + inc)
        for count_cat in range(num_cats):
            passed = []
            for cut_cat in range(num_cats):
                passed.append(
                    events[
                        (events["cut"] == 0)
                        & (events[cat_name] == cut_cat)
                        & (events[prefix + str(count_cat)] > cuts[cut + 1])
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

            if (
                sig_effs[count_cat][cut + 1] == 0.0
                or bkg_effs[count_cat][cut + 1] == 0.0
            ):
                purities[count_cat].append(0.0)
                foms[count_cat].append(0.0)
            else:
                purities[count_cat].append(passed[count_cat] / sum(passed))
                foms[count_cat].append(
                    sig_effs[count_cat][cut + 1] * purities[count_cat][cut + 1]
                )

            if foms[count_cat][cut + 1] > max_foms[count_cat]:
                max_foms[count_cat] = foms[count_cat][cut + 1]
                max_fom_cuts[count_cat] = cuts[cut + 1]

    # Convert the lists to numpy arrays
    cuts = np.asarray(cuts)
    sig_effs = np.asarray(sig_effs)
    bkg_effs = np.asarray(bkg_effs)
    purities = np.asarray(purities)
    foms = np.asarray(foms)

    # Calculate summary values
    sig_effs_auc = []
    bkg_effs_auc = []
    pur_auc = []
    fom_auc = []
    roc_auc = []
    for cat in range(num_cats):
        sig_effs_auc.append(auc(cuts, sig_effs[cat]))
        bkg_effs_auc.append(auc(cuts, bkg_effs[cat]))
        pur_auc.append(auc(cuts, purities[cat]))
        fom_auc.append(auc(cuts, foms[cat]))
        roc_auc.append(auc(bkg_effs[cat], sig_effs[cat]))

    # Print summary if verbose
    if verbose:
        for cat in range(num_cats):
            label = chipsnet.data.get_map(cat_name)["labels"][cat]
            print(label + ": {0:.4f}({1:.4f})".format(max_foms[cat], max_fom_cuts[cat]))

        print("Signal efficiency AUC: {}".format(sig_effs_auc))
        print("Background efficiency AUC: {}".format(bkg_effs_auc))
        print("Purity AUC: {}".format(pur_auc))
        print("FOM AUC: {}".format(fom_auc))
        print("ROC AUC: {}".format(roc_auc))

    # We now use the maximum figure-of-merit values to calculate the efficiency
    # and purity hists with neutrino energy.
    e_bins = 14
    e_range = (1000, 8000)
    fom_effs = []
    fom_purs = []
    for count_cat in range(num_cats):
        signal_h = None
        bkg_h = np.zeros(e_bins)
        eff_hists = []
        for cut_cat in range(num_cats):
            total = events[events[cat_name] == cut_cat]
            passed = events[
                (events[cat_name] == cut_cat)
                & (events["cut"] == 0)
                & (events[prefix + str(count_cat)] > max_fom_cuts[count_cat])
            ]

            total_h = np.histogram(
                total["t_nuEnergy"], bins=e_bins, range=e_range, weights=total["w"]
            )
            passed_h = np.histogram(
                passed["t_nuEnergy"], bins=e_bins, range=e_range, weights=passed["w"]
            )
            eff_h = np.divide(passed_h, total_h)
            eff_hists.append(eff_h[0])

            if cut_cat == count_cat:
                signal_h = passed_h[0]
            else:
                bkg_h = np.add(bkg_h, passed_h[0])

        fom_effs.append(eff_hists)
        fom_purs.append(
            np.divide(
                signal_h,
                np.add(signal_h, bkg_h),
                out=np.zeros_like(bkg_h),
                where=(bkg_h != 0),
            )
        )

    output = {
        "cuts": cuts,
        "sig_effs": sig_effs,
        "bkg_effs": bkg_effs,
        "purs": purities,
        "foms": foms,
        "fom_effs": fom_effs,
        "fom_purs": fom_purs,
        "sig_effs_auc": sig_effs_auc,
        "bkg_effs_auc": bkg_effs_auc,
        "pur_auc": pur_auc,
        "fom_auc": fom_auc,
        "roc_auc": roc_auc,
    }

    return output


def classify(event, categories, prefix):
    """Classify the event by the highest score.
    Args:
        event (dict): Pandas event(row) dict
        categories (int): Number of categories
        prefix: (str): prefix of the different scores for this classification
    Returns:
        int: category classification
    """
    if categories == 1:
        return round(event[prefix])
    else:
        x = [event[prefix + str(i)] for i in range(categories)]
        return np.asarray(x).argmax()


def run_pca(
    events,
    model,
    layer_name="dense_final",
    standardise=True,
    components=3,
    verbose=False,
):
    """Run PCA on the final dense layer outputs and append results to events dataframe.
    Args:
        events (pandas.DataFrame): Events dataframe to append outputs
        model (chipsnet.Model): Model to use for prediction
        layer_name (str): Final dense layer name
        standardise (bool): Should we apply a standard scalar to the dense layer outputs?
        components (int): Number of PCA componenets to calculate
        verbose (bool): Should we print summary?
    Returns:
        pandas.DataFrame: Events dataframe with PCA outputs
    """
    # Create model that outputs the dense layer outputs
    dense_model = Model(
        inputs=model.model.input, outputs=model.model.get_layer(layer_name).output
    )

    # Make the predictions
    dense_outputs = dense_model.predict(np.stack(events["image_0"].to_numpy()))

    # Run a standard scalar on the dense outputs if needed
    if standardise:
        dense_outputs = StandardScaler().fit_transform(dense_outputs)

    # Run the PCA
    pca = PCA(n_components=components)
    pca_result = pca.fit_transform(dense_outputs)
    pca_result = pd.DataFrame(pca_result)

    # Append the results to the events dataframe
    for component in range(components):
        events["pca_" + str(component)] = pca_result[component]

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
    layer_name="dense_final",
    standardise=True,
    components=3,
    max_events=10000,
):
    """Run t-SNE on the final dense layer outputs and append results to events dataframe.
    Args:
        events (pandas.DataFrame): Events dataframe to append outputs
        model (chipsnet.Model): Model to use for prediction
        layer_name (str): Final dense layer name
        standardise (bool): Should we apply a standard scalar to the dense layer outputs?
        components (int): Number of t-SNE componenets to calculate
        verbose (bool): Should we print summary?
    Returns:
        pandas.DataFrame: Events dataframe with t-SNE outputs
    """
    # Create model that outputs the dense layer outputs
    dense_model = Model(
        inputs=model.model.input, outputs=model.model.get_layer(layer_name).output
    )

    # Make the predictions
    dense_outputs = dense_model.predict(np.stack(events["image_0"].to_numpy()))

    # Run a standard scalar on the dense outputs if needed
    if standardise:
        dense_outputs = StandardScaler().fit_transform(dense_outputs)

    # Run t-SNE
    tsne = TSNE(n_components=components, verbose=1, perplexity=40, n_iter=300)
    tsne_result = tsne.fit_transform(dense_outputs[:max_events])
    tsne_result = pd.DataFrame(tsne_result)

    # Append the results to the events dataframe
    for component in range(components):
        events["tsne_" + str(component)] = tsne_result[component]

    return events


def explain_gradcam(
    events, model, num_events, output="t_all_cat", layer_name="path0_block1"
):
    """Run GradCAM on the given model using the events.
    Args:
        events (pandas.DataFrame): Events dataframe to append outputs
        model (chipsnet.Model): Model to use for prediction
        num_events (int): Number of events to run
        output (str): Single classification model output to use
        layer_name (str): Model layer name to use
    Returns:
        list[event_outputs]: List of GradCAM outputs
    """
    explain_m = Model(
        inputs=model.model.input, outputs=model.model.get_layer(output).output
    )
    outputs = []
    for event in range(num_events):
        category = int(events["t_all_cat"][event])
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
        events (pandas.DataFrame): Events dataframe to append outputs
        model (chipsnet.Model): Model to use for prediction
        num_events (int): Number of events to run
        output (str): Single classification model output to use
    Returns:
        list[event_outputs]: List of OcclusionSensitivity outputs
    """
    explain_m = Model(
        inputs=model.model.input, outputs=model.model.get_layer(output).output
    )
    outputs = []
    for event in range(num_events):
        category = int(events["t_all_cat"][event])
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
        events (pandas.DataFrame): Events dataframe to append outputs
        model (chipsnet.Model): Model to use for prediction
        num_events (int): Number of events to run
        output (str): Single classification model output to use
        layer_name (str): Model layer name to use
    Returns:
        list[event_outputs]: List of ExtractActivations outputs
    """
    explain_m = Model(
        inputs=model.model.input, outputs=model.model.get_layer(output).output
    )
    outputs = []
    for event in range(num_events):
        category = int(events["t_all_cat"][event])
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
        events (pandas.DataFrame): Events dataframe to append outputs
        model (chipsnet.Model): Model to use for prediction
        num_events (int): Number of events to run
        output (str): Single classification model output to use
    Returns:
        dict: Dictionary of gradient outputs
    """
    explain_m = Model(
        inputs=model.model.input, outputs=model.model.get_layer(output).output
    )
    outputs = {"vanilla": [], "smooth": [], "integrated": [], "inputs": []}
    for event in range(num_events):
        category = int(events["t_all_cat"][event])
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


def predict_energies(self):
    """Estimate the event energy using the true category model
    """
    estimations = []
    for i in tqdm(
        range(self.beam_model.categories + 1), desc="--- predicting energies"
    ):
        estimations.append(self.energy_models[i].model.predict(self.data))

    true_cat_energies = []
    pred_cat_energies = []
    for i in range(self.events.shape[0]):
        true_cat_energies.append(estimations[self.events["t_cat"][i]][i])
        pred_cat_energies.append(estimations[self.events["classification"][i]][i])

    self.events["true_cat_pred_e"] = np.array(true_cat_energies)
    self.events["pred_cat_pred_e"] = np.array(pred_cat_energies)


def globes_smearing_file(hists, names):
    """Output Globes compatible energy smearing files.
    Args:
        hists (list[numpy 2darray]): List of histogram energy smearing arrays
        names (list[str]): Event type name
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
