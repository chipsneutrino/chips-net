# -*- coding: utf-8 -*-

"""Utility module containing lots of helpful methods for evaluation and plotting
"""

import time
import copy

import pandas as pd
import numpy as np
from tensorflow.keras import Model
from tqdm import tqdm
from root_pandas import to_root

import chipsnet.config
import chipsnet.data as data
import chipsnet.models


def standard_model_evaluation(data, model):
    """Run a standard evaluation on the model
    Args:
        data (tf.dataset): Input dataset
        model (chipsnet.Model): Model to use for prediction
    """
    print('\n--- Running Evaluation ---')
    start_time = time.time()  # Time how long it takes
    '''
    self.load_models()  # Load all the required models
    self.load_dataset()  # Load the dataset into the pandas df
    self.calculate_weights()  # Calculate the weights to apply categorically
    self.run_inference()  # Get model outputs for test data
    self.calculate_cuts()  # Calculate different cuts to be applied
    print('--- classifying events... ', end='', flush=True)
    self.events['classification'] = self.events.apply(self.classify, axis=1)
    print('done')
    self.predict_energies()
    self.save_to_file()  # Save everything to a ROOT file
    '''
    print('--- Done (took %s seconds) ---\n' % (time.time() - start_time))


def data_from_conf(config, name):
    """Get the input data from the configuration.
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
    """Get and load the model from the configuration.
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
    chipsnet.config.setup_dirs(model_config, False)
    model = chipsnet.models.get_model(model_config)
    model.load()
    return model


def run_inference(data, events, model, num_events, prefix=''):
    """Run predictions on the input dataset and append outputs to events dataframe.
    Args:
        data (tf.dataset): Input dataset
        events (pandas.DataFrame): Events dataframe to append outputs
        model (chipsnet.Model): Model to use for prediction
        num_events (int): Number of events to predict
        prefix (str): Prefix to append to column name
    Returns:
        pandas.DataFrame: Events dataframe with model predictions
    """
    print('Running inference on {}... '.format(model.model.name), end='', flush=True)
    start_time = time.time()
    data = data.testing_ds(num_events, batch_size=64)
    num_model_outputs = len(model.config.model.labels)
    outputs = model.model.predict(data)
    for i, label in enumerate(model.config.model.labels):
        base_key = prefix + 'pred_' + label
        if int(num_model_outputs) == 1:
            events[base_key] = outputs
        elif outputs[i].shape[1] == 1:
            events[base_key] = outputs[i]
        else:
            for j in range(outputs[i].shape[1]):
                key = base_key + '_' + str(j)
                events[key] = outputs[i][:, j]
    print('took {:.2f} seconds'.format(time.time() - start_time))
    return events


def full_comb_combine(events, prefix):
    """Get the dense layer outputs for the specified dataset and model.
    Args:
        events (pandas.DataFrame): Events dataframe to calculate combined category scores
        prefix (str): Prefix to apply to model output values
    Returns:
        pandas.DataFrame: Events dataframe with combined category scores
    """
    def full_comb_apply(event, prefix):
        nuel_cc_cats = [0, 1, 2, 3, 4, 5]
        nuel_cc_value = 0.0
        for cat in nuel_cc_cats:
            nuel_cc_value = nuel_cc_value + event[prefix + str(cat)]

        numu_cc_cats = [12, 13, 14, 15, 16, 17]
        numu_cc_value = 0.0
        for cat in numu_cc_cats:
            numu_cc_value = numu_cc_value + event[prefix + str(cat)]

        nc_cats = [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]
        nc_value = 0.0
        for cat in nc_cats:
            nc_value = nc_value + event[prefix + str(cat)]

        return nuel_cc_value, numu_cc_value, nc_value

    apply_prefix = prefix + "pred_t_all_cat_"
    comb_prefix = prefix + "pred_t_comb_cat_"
    events["scores"] = events.apply(full_comb_apply, axis=1, args=(apply_prefix,))
    events[comb_prefix + "0"] = events.scores.map(lambda x: x[0])
    events[comb_prefix + "1"] = events.scores.map(lambda x: x[1])
    events[comb_prefix + "2"] = events.scores.map(lambda x: x[2])
    events.drop(["scores"], axis=1)
    return events


def dense_output(data, model, layer_name="dense_final"):
    """Get the dense layer outputs for the specified dataset and model.
    Args:
        data (tf.dataset): Input dataset
        model (chipsnet.Model): Model to use
    Returns:
        np.array: Array of dense layer outputs
    """
    # Create model that outputs the dense layer outputs
    dense_model = Model(
        inputs=model.model.input,
        outputs=model.model.get_layer(layer_name).output
    )
    return dense_model.predict(data)


def apply_weights(
    events,
    total_num=1214165.85244438,   # for chips_1200
    nuel_frac=0.00003202064566,   # for chips_1200
    anuel_frac=0.00000208200747,  # for chips_1200
    numu_frac=0.00276174709613,   # for chips_1200
    anumu_frac=0.00006042213136,  # for chips_1200
    cosmic_frac=0.99714372811940  # for chips_1200
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
        if (event[data.MAP_NU_TYPE.name] == 0 and
                event[data.MAP_SIGN_TYPE.name] == 0 and
                event[data.MAP_COSMIC_CAT.name] == 0):
            return w_nuel
        elif (event[data.MAP_NU_TYPE.name] == 0 and
                event[data.MAP_SIGN_TYPE.name] == 1 and
                event[data.MAP_COSMIC_CAT.name] == 0):
            return w_anuel
        elif (event[data.MAP_NU_TYPE.name] == 1 and
                event[data.MAP_SIGN_TYPE.name] == 0 and
                event[data.MAP_COSMIC_CAT.name] == 0):
            return w_numu
        elif (event[data.MAP_NU_TYPE.name] == 1 and
                event[data.MAP_SIGN_TYPE.name] == 1 and
                event[data.MAP_COSMIC_CAT.name] == 0):
            return w_anumu
        elif (event[data.MAP_COSMIC_CAT.name] == 1):
            return w_cosmic
        else:
            raise NotImplementedError

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

    w_nuel = (1.0/tot_nuel)*(nuel_frac * total_num)
    w_anuel = (1.0/tot_anuel)*(anuel_frac * total_num)
    w_numu = (1.0/tot_numu)*(numu_frac * total_num)
    w_anumu = (1.0/tot_anumu)*(anumu_frac * total_num)
    w_cosmic = (1.0/tot_cosmic)*(cosmic_frac * total_num)

    events['w'] = events.apply(
        add_weight,
        axis=1,
        args=(w_nuel, w_anuel, w_numu, w_anumu, w_cosmic)
    )

    print("Nuel number:   {}, weight: {:.5f}".format(tot_nuel, w_nuel))
    print("Anuel number:  {}, weight: {:.5f}".format(tot_anuel, w_anuel))
    print("Numu number:   {}, weight: {:.5f}".format(tot_numu, w_numu))
    print("Anumu number:  {}, weight: {:.5f}".format(tot_anumu, w_anumu))
    print("Cosmic number: {}, weight: {:.3f}".format(tot_cosmic, w_cosmic))
    return events


def apply_standard_cuts(
    events,
    cosmic_cut=0.001,
    q_cut=500.0,
    h_cut=500.0,
    theta_cut=0.7,
    phi_cut=0.3
        ):
    """Calculate and apply the standard cuts to the events dataframe.
    Args:
        events (pandas.DataFrame): Events dataframe to append weights to
        cosmic_cut (float): Cosmic classifier output cut
        q_cut (float): Total collected event chargr cut
        h_cut (float): Hough peak height cut
        theta_cut (float): Reco theta direction cut
        phi_cut (float): Reco phi direction cut
    Returns:
        events (pandas.DataFrame): Events with cuts applied
    """
    cosmic_cut_func = cut_apply("pred_t_cosmic_cat", cosmic_cut, type='greater')
    cosmic_cuts = events.apply(cosmic_cut_func, axis=1)

    q_cut_func = cut_apply('r_first_ring_height', h_cut, type='lower')
    q_cuts = events.apply(q_cut_func, axis=1)

    h_cut_func = cut_apply('r_raw_total_digi_q', q_cut, type='lower')
    h_cuts = events.apply(h_cut_func, axis=1)

    theta_low_cut_func = cut_apply('r_dirTheta', -theta_cut, type='lower')
    theta_low_cuts = events.apply(theta_low_cut_func, axis=1)

    theta_high_cut_func = cut_apply('r_dirTheta', theta_cut, type='greater')
    theta_high_cuts = events.apply(theta_high_cut_func, axis=1)

    phi_low_cut_func = cut_apply('r_dirPhi', -phi_cut, type='lower')
    phi_low_cuts = events.apply(phi_low_cut_func, axis=1)

    phi_high_cut_func = cut_apply('r_dirPhi', phi_cut, type='greater')
    phi_high_cuts = events.apply(phi_high_cut_func, axis=1)

    events["cut"] = np.logical_or.reduce((cosmic_cuts, q_cuts, h_cuts,
                                          theta_low_cuts, theta_high_cuts,
                                          phi_low_cuts, phi_high_cuts))

    for i in range(len(data.MAP_FULL_COMB_CAT.labels)):
        cat_events = events[events[data.MAP_FULL_COMB_CAT.name] == i]
        survived_events = cat_events[cat_events["cut"] == 0]
        print("{}: total {}, survived: {}".format(
            data.MAP_FULL_COMB_CAT.labels[i],
            cat_events.shape[0],
            survived_events.shape[0]/cat_events.shape[0])
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
        if type == 'greater':
            return (event[variable] >= value)
        elif type == 'lower':
            return (event[variable] <= value)
        else:
            raise NotImplementedError
    return cut_func


def calculate_curves(events, cat_name='t_comb_cat', thresholds=200, prefix=""):
    """Calculate efficiency curves etc...
    Args:
        events (pandas.DataFrame): Events dataframe to use
        cat_name (str): Category name to use
        thresholds (int): Number of threshold values to use
        prefix (str): Prefix to apply to model output values
    Returns:
        cuts (np.array): Array of threshold cut values
        sig_effs (np.array): Array signal efficiencies
        bkg_effs (np.array): Array background efficiencies
        purities (np.array): Array of purities
        foms (np.array): Array of figure-of-merits (efficiency*purity)
    """
    num_cats = chipsnet.data.get_map(cat_name).categories
    prefix = prefix + "pred_" + cat_name + "_"

    cuts, totals, max_foms, max_fom_cuts = [], [], [], []
    sig_effs, bkg_effs, purities, foms = [], [], [], []
    for cat in range(num_cats):
        totals.append(events[events[cat_name] == cat]['w'].sum())
        max_foms.append(0.0)
        max_fom_cuts.append(0.0)

    for cat in range(num_cats):
        sig_effs.append([1.0])
        bkg_effs.append([1.0])
        purities.append([totals[cat]/sum(totals)])
        foms.append([sig_effs[cat][0]*purities[cat][0]])

    cuts.append(0.0)
    inc = float(1.0/thresholds)
    for cut in tqdm(range(thresholds), desc='--- calculating curves'):
        cuts.append((cut * inc) + inc)
        for count_cat in range(num_cats):
            passed = []
            for cut_cat in range(num_cats):
                passed.append(events[(events[cat_name] == cut_cat) &
                                     (events[prefix + str(count_cat)] > cuts[cut+1])]['w'].sum())

            # Calculate the signal and background efficiencies for this category
            if passed[count_cat] == 0.0 or totals[count_cat] == 0.0:
                sig_effs[count_cat].append(0.0)
            else:
                sig_effs[count_cat].append(passed[count_cat]/totals[count_cat])
            bkg_passed = sum(passed[:count_cat]+passed[count_cat+1:])
            bkg_total = sum(totals[:count_cat]+totals[count_cat+1:])
            if bkg_passed == 0.0 or bkg_total == 0.0:
                bkg_effs[count_cat].append(0.0)
            else:
                bkg_effs[count_cat].append(bkg_passed/bkg_total)

            if sig_effs[count_cat][cut+1] == 0.0 or bkg_effs[count_cat][cut+1] == 0.0:
                purities[count_cat].append(0.0)
                foms[count_cat].append(0.0)
            else:
                purities[count_cat].append(passed[count_cat]/sum(passed))
                foms[count_cat].append(sig_effs[count_cat][cut+1]*purities[count_cat][cut+1])

            if foms[count_cat][cut+1] > max_foms[count_cat]:
                max_foms[count_cat] = foms[count_cat][cut+1]
                max_fom_cuts[count_cat] = cuts[cut+1]

    print("--- Maximum FOMS ---")
    for cat in range(num_cats):
        label = chipsnet.data.get_map(cat_name).labels[cat]
        print(label + ": {0:.4f}({1:.4f})".format(max_foms[cat], max_fom_cuts[cat]))

    # Convert the lists to numpy arrays
    cuts = np.asarray(cuts)
    sig_effs = np.asarray(sig_effs)
    bkg_effs = np.asarray(bkg_effs)
    purities = np.asarray(purities)
    foms = np.asarray(foms)

    # Calculate the areas under the curves and print them
    sig_effs_int = np.trapz(sig_effs[0], x=cuts)
    bkg_effs_int = np.trapz(bkg_effs[0], x=cuts)
    purities_int = np.trapz(purities[0], x=cuts)
    fom_int = np.trapz(foms[0], x=cuts)
    sig_vs_bkg_int = np.trapz(sig_effs[0], x=bkg_effs[0])
    print("Signal efficiency AUC: {}".format(sig_effs_int))
    print("Background efficiency AUC: {}".format(bkg_effs_int))
    print("Purity AUC: {}".format(purities_int))
    print("FOM AUC: {}".format(fom_int))
    print("ROC AUC: {}".format(sig_vs_bkg_int))

    return cuts, sig_effs, bkg_effs, purities, foms


def classify(event, categories, prefix):
    """Add the correct weight to each event.
    Args:
        event (dict): Pandas event(row) dict
    Returns:
        int: category classification
    """
    if categories == 1:
        return round(event[prefix])
    else:
        x = [event[prefix + "_" + str(i)] for i in range(categories)]
        return np.asarray(x).argmax()


def predict_energies(self):
    """Estimate the event energy using the true category model
    """
    estimations = []
    for i in tqdm(range(self.beam_model.categories+1), desc='--- predicting energies'):
        estimations.append(self.energy_models[i].model.predict(self.data))

    true_cat_energies = []
    pred_cat_energies = []
    for i in range(self.events.shape[0]):
        true_cat_energies.append(estimations[self.events['t_cat'][i]][i])
        pred_cat_energies.append(estimations[self.events['classification'][i]][i])

    self.events['true_cat_pred_e'] = np.array(true_cat_energies)
    self.events['pred_cat_pred_e'] = np.array(pred_cat_energies)


def globes_smearing_file(hists, names):
    """Output Globes compatible energy smearing files.
    Args:
        hists (list[numpy 2darray]): List of histogram energy smearing arrays
        names (list[str]): Event type name
    """
    for hist, name in zip(hists, names):
        with open("smearing/" + name + ".dat", 'w', encoding='utf-8') as f:
            f.write("energy(#" + name + ")<\n")
            f.write("@energy =\n")
            for i in range(40):
                f.write("{0, 40, ")
                for j in range(40):
                    if j == 39:
                        f.write(str.format('{0:.5f}', hist[j, i]))
                    else:
                        f.write(str.format('{0:.5f}, ', hist[j, i]))
                f.write("}:\n")
            f.write(">\n")


def save_to_file(events, path):
    """Save events dataframe to a ROOT file.
    Args:
        events (pandas.DataFrame): Events dataframe to save
        names (str): Output file path
    """
    to_root(events, path, key='events')
