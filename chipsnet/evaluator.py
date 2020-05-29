# -*- coding: utf-8 -*-

"""Evaluator classes for testing the trained models

This module contains all the evaluation classes used to test the
trained models on test data to determine their performance. All are
derived from the BaseEvaluator class.
"""

import time

import pandas as pd
import numpy as np
from tensorflow.keras import Model
from tqdm import tqdm

import chipsnet.config
import chipsnet.data
import chipsnet.models


class Evaluator(object):
    """Combined Cosmic and Beam classification model evaluation.
    """

    def __init__(self, config):
        """Initialise the Evaluator.
        Args:
            config (dotmap.DotMap): DotMap Configuration namespace
        """
        super().__init__(config)
        self.init_evaluator()

    def init_evaluator(self):
        """Initialise the evaluator.
        """
        # Get the test dataset
        data_reader = chipsnet.data.Reader(self.config)
        self.data = data_reader.test_data()
        self.data = self.data.take(self.config.eval.test_examples)
        self.data = self.data.batch(self.config.eval.batch_size, drop_remainder=True)

        # Fully combined category names
        self.comb_cat_names = ['Nuel-CC', 'Numu-CC', 'NC', 'Cosmic']

    def get_loaded_model(self, name):
        """Get and loads the model from the configuration.
        Args:
            name (str): Saved model name
        Returns:
            chipscnv.models model: Model to use in evaluation
        """
        config = self.config
        config.model = self.config.models[name]
        config.exp.output_dir = self.config.models[name].dir
        config.exp.name = self.config.models[name].path
        chipsnet.config.setup_dirs(config, False)
        model = chipsnet.models.get_model(config)
        model.load()
        return model

    def run_all(self):
        """Run the full evaluation.
        """
        print('\n--- Running Evaluation ---')

        start_time = time.time()  # Time how long it takes

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

        print('--- Done (took %s seconds) ---\n' % (time.time() - start_time))

    def load_models(self):
        """Load all the required models.
        """
        self.beam_model = self.get_loaded_model("beam_classification")
        with tqdm(total=(self.beam_model.categories+3), desc='--- loading models') as pbar:
            pbar.update()
            self.cosmic_model = self.get_loaded_model("cosmic_classification")
            pbar.update()
            # Setup all the energy estimation models
            self.energy_models = []
            for i in range(self.beam_model.categories):
                name = "cat_" + str(i) + "_energy"
                self.energy_models.append(self.get_loaded_model(name))
                pbar.update()
            self.energy_models.append(self.get_loaded_model("cosmic_energy"))
            pbar.update()

    def load_dataset(self):
        """Load the dataset into the pandas df.
        """
        print('--- loading dataset... ', end='', flush=True)

        events = {  # Create empty dict to hold all the event data
            't_nu': [], 't_code': [], 't_cat': [], 't_cosmic_cat': [],
            't_full_cat': [], 't_nu_nc_cat': [], 't_nc_cat': [],
            't_vtxX': [], 't_vtxY': [], 't_vtxZ': [], 't_vtxT': [], 't_nuEnergy': [],
            't_p_pdgs': [], 't_p_energies': [], 't_p_dirTheta': [], 't_p_dirPhi': [],
            'r_raw_num_hits': [], 'r_filtered_num_hits': [], 'r_num_hough_rings': [],
            'r_raw_total_digi_q': [], 'r_filtered_total_digi_q': [],
            'r_first_ring_height': [], 'r_last_ring_height': [],
            'r_vtxX': [], 'r_vtxY': [], 'r_vtxZ': [], 'r_vtxT': [],
            'r_dirTheta': [], 'r_dirPhi': []
        }

        images = self.config.data.img_size[2]
        if self.config.data.stack:
            images = 1
        for i in range(images):
            events[('image_' + str(i))] = []

        for x, y in self.data:
            for name, array in list(x.items()):  # Fill events dict with 'inputs'
                if name in events.keys():
                    events[name].extend(array.numpy())

            for name, array in list(y.items()):  # Fill events dict with 'labels'
                if name in events.keys():
                    events[name].extend(array.numpy())

        self.events = pd.DataFrame.from_dict(events)  # Convert dict to pandas dataframe
        print('done')

    def calculate_weights(self):
        """Calculate the weights to apply categorically.
        """
        print('--- calculating weights... ', end='', flush=True)

        tot_nuel = self.events[(self.events.t_nu == 0) &
                               (self.events.t_cosmic_cat == 0)].shape[0]
        tot_numu = self.events[(self.events.t_nu == 1) &
                               (self.events.t_cosmic_cat == 0)].shape[0]
        tot_cosmic = self.events[self.events.t_cosmic_cat == 1].shape[0]

        self.nuel_weight = (1.0/tot_nuel)*(self.config.eval.weights.nuel *
                                           self.config.eval.weights.total)
        self.numu_weight = (1.0/tot_numu)*(self.config.eval.weights.numu *
                                           self.config.eval.weights.total)
        self.cosmic_weight = (1.0/tot_cosmic)*(self.config.eval.weights.cosmic *
                                               self.config.eval.weights.total)

        self.events['weight'] = self.events.apply(self.add_weight, axis=1)

        print("Nuel: {0:.4f}({1}), Numu: {2:.4f}({3}), Cosmic: {4:.4f}({5})".format(
            self.nuel_weight, tot_nuel,
            self.numu_weight, tot_numu,
            self.cosmic_weight, tot_cosmic))

    def add_weight(self, event):
        """Add the correct weight to each event.
        Args:
            event (dict): Pandas event(row) dict
        Returns:
            float: Weight to use for event
        """
        if event.t_nu == 0 and event.t_cosmic_cat == 0:
            return self.nuel_weight
        elif event.t_nu == 1 and event.t_cosmic_cat == 0:
            return self.numu_weight
        elif event.t_cosmic_cat == 1:
            return self.cosmic_weight
        else:
            raise NotImplementedError

    def run_inference(self):
        """Get model outputs for test data.
        """
        print('--- running inference... ', end='', flush=True)
        c_dense_model = None
        b_dense_model = None
        if self.config.eval.make_dense:
            c_dense_model = Model(  # Model to ouput cosmic model dense layer
                inputs=self.cosmic_model.model.input,
                outputs=self.cosmic_model.model.get_layer('dense_final').output
            )
            b_dense_model = Model(  # Model to ouput beam model dense layer
                inputs=self.beam_model.model.input,
                outputs=self.beam_model.model.get_layer('dense_final').output
            )

        self.events['c_out'] = self.cosmic_model.model.predict(self.data)
        beam_predictions = self.beam_model.model.predict(self.data)
        for i in range(self.beam_model.categories):
            self.events[('b_out_' + str(i))] = beam_predictions[:, i]
        if self.config.eval.make_dense:
            self.c_dense = c_dense_model.predict(self.data)
            self.b_dense = b_dense_model.predict(self.data)

        # self.events.c_out = self.events.c_out.map(lambda x: print(x.shape))
        self.events['scores'] = self.events.apply(self.beam_model.combine_outputs, axis=1)
        self.events['nuel_score'] = self.events.scores.map(lambda x: x[0])
        self.events['numu_score'] = self.events.scores.map(lambda x: x[1])
        self.events['nc_score'] = self.events.scores.map(lambda x: x[2])
        self.events.drop('scores', axis=1, inplace=True)
        print('done')

    def calculate_cuts(self):
        """Calculate different cuts to be applied.
        """
        print('--- calculating cuts... ', end='', flush=True)

        self.events['base_cut'] = self.events.apply(self.base_cut, axis=1)
        self.events['cosmic_cut'] = self.events.apply(self.cosmic_cut, axis=1)

        events = self.events[(self.events.base_cut == 0) & (self.events.cosmic_cut == 0)]
        nuel_tot = events[events.t_full_cat == 0]['weight'].sum()
        numu_tot = events[events.t_full_cat == 1]['weight'].sum()
        nc_tot = events[events.t_full_cat == 2]['weight'].sum()

        cut = []
        nuel_eff_sig, nuel_eff_bkg, = [], []
        nuel_pur, nuel_fom = [], []
        numu_eff_sig, numu_eff_bkg,  = [], []
        numu_pur, numu_fom = [], []
        nc_eff_sig, nc_eff_bkg,  = [], []
        nc_pur, nc_fom = [], []

        bins = 100
        self.nuel_max_fom = 0.0
        self.numu_max_fom = 0.0
        self.nc_max_fom = 0.0
        self.nuel_max_fom_cut = 0
        self.numu_max_fom_cut = 0
        self.nc_max_fom_cut = 0

        cut.append(0.0)
        nuel_eff_sig.append(1.0)
        numu_eff_sig.append(1.0)
        nc_eff_sig.append(1.0)
        nuel_eff_bkg.append(1.0)
        numu_eff_bkg.append(1.0)
        nc_eff_bkg.append(1.0)
        nuel_pur.append(nuel_tot/(nuel_tot+numu_tot+nc_tot))
        numu_pur.append(numu_tot/(nuel_tot+numu_tot+nc_tot))
        nc_pur.append(nc_tot/(nuel_tot+numu_tot+nc_tot))
        nuel_fom.append(nuel_eff_sig[0]*nuel_pur[0])
        numu_fom.append(numu_eff_sig[0]*numu_pur[0])
        nc_fom.append(nc_eff_sig[0]*nc_pur[0])

        for bin in range(bins):
            cut.append((bin * 0.01) + 0.01)

            nuel_nuel_cut = events[(events.t_full_cat == 0) &
                                   (events.nuel_score > cut[bin+1])]['weight'].sum()
            numu_nuel_cut = events[(events.t_full_cat == 1) &
                                   (events.nuel_score > cut[bin+1])]['weight'].sum()
            nc_nuel_cut = events[(events.t_full_cat == 2) &
                                 (events.nuel_score > cut[bin+1])]['weight'].sum()

            nuel_numu_cut = events[(events.t_full_cat == 0) &
                                   (events.numu_score > cut[bin+1])]['weight'].sum()
            numu_numu_cut = events[(events.t_full_cat == 1) &
                                   (events.numu_score > cut[bin+1])]['weight'].sum()
            nc_numu_cut = events[(events.t_full_cat == 2) &
                                 (events.numu_score > cut[bin+1])]['weight'].sum()

            nuel_nc_cut = events[(events.t_full_cat == 0) &
                                 (events.nc_score > cut[bin+1])]['weight'].sum()
            numu_nc_cut = events[(events.t_full_cat == 1) &
                                 (events.nc_score > cut[bin+1])]['weight'].sum()
            nc_nc_cut = events[(events.t_full_cat == 2) &
                               (events.nc_score > cut[bin+1])]['weight'].sum()

            nuel_eff_sig.append(nuel_nuel_cut/nuel_tot)
            numu_eff_sig.append(numu_numu_cut/numu_tot)
            nc_eff_sig.append(nc_nc_cut/nc_tot)

            nuel_eff_bkg.append((numu_nuel_cut+nc_nuel_cut)/(numu_tot+nc_tot))
            numu_eff_bkg.append((nuel_numu_cut+nc_numu_cut)/(nuel_tot+nc_tot))
            nc_eff_bkg.append((nuel_nc_cut+numu_nc_cut)/(nuel_tot+numu_tot))

            if nuel_eff_sig[bin+1] == 0.0 or nuel_eff_bkg[bin+1] == 0.0:
                nuel_pur.append(0.0)
                nuel_fom.append(0.0)
            else:
                nuel_pur.append(nuel_nuel_cut/(nuel_nuel_cut+numu_nuel_cut+nc_nuel_cut))
                nuel_fom.append(nuel_eff_sig[bin]*nuel_pur[bin])

            if numu_eff_sig[bin+1] == 0.0 or numu_eff_bkg[bin+1] == 0.0:
                numu_pur.append(0.0)
                numu_fom.append(0.0)
            else:
                numu_pur.append(numu_numu_cut/(nuel_numu_cut+numu_numu_cut+nc_numu_cut))
                numu_fom.append(numu_eff_sig[bin]*numu_pur[bin])

            if nc_eff_sig[bin+1] == 0.0 or nc_eff_bkg[bin+1] == 0.0:
                nc_pur.append(0.0)
                nc_fom.append(0.0)
            else:
                nc_pur.append(nc_nc_cut/(nuel_nc_cut+numu_nc_cut+nc_nc_cut))
                nc_fom.append(nc_eff_sig[bin]*nc_pur[bin])

            if nuel_fom[bin] > self.nuel_max_fom:
                self.nuel_max_fom = nuel_fom[bin]
                self.nuel_max_fom_cut = cut[bin]

            if numu_fom[bin] > self.numu_max_fom:
                self.numu_max_fom = numu_fom[bin]
                self.numu_max_fom_cut = cut[bin]

            if nc_fom[bin] > self.nc_max_fom:
                self.nc_max_fom = nc_fom[bin]
                self.nc_max_fom_cut = cut[bin]

        print("Nuel: {0:.4f}({1:.4f}), Numu: {2:.4f}({3:.4f}), NC: {4:.4f}({5:.4f})".format(
            self.nuel_max_fom, self.nuel_max_fom_cut,
            self.numu_max_fom, self.numu_max_fom_cut,
            self.nc_max_fom, self.nc_max_fom_cut))

    def base_cut_summary(self):
        """Print how each category is affected by the base_cut.
        """
        print("Base Cut Summary...\n")
        for i in range(4):
            cat_events = self.events[self.events.t_full_cat == i]
            print("{}-> Total {}, Survived: {}\n".format(
                self.comb_cat_names[i], cat_events.shape[0],
                cat_events[cat_events['base_cut'] == 0].shape[0]/cat_events.shape[0]))

    def combined_cut_summary(self):
        """Print how each category is affected by the base_cut + cosmic_cut.
        """
        print("Base + Cosmic Cut Summary...\n")
        for i in range(4):
            cat_events = self.events[self.events.t_full_cat == i]
            print("{}-> Total {}, Survived: {}\n".format(
                self.comb_cat_names[i], cat_events.shape[0], cat_events[
                    (cat_events.base_cut == 0) &
                    (cat_events.cosmic_cut == 0)].shape[0]/cat_events.shape[0]))

    def base_cut(self, event):
        """Calculate if the event should be cut due to activity.
        Args:
            event (dict): Pandas event(row) dict
        Returns:
            bool: cut or not?
        """
        cut = ((event.r_raw_total_digi_q <= self.config.eval.cuts.q) or
               (event.r_first_ring_height <= self.config.eval.cuts.hough) or
               (event.r_dirTheta <= -self.config.eval.cuts.theta) or
               (event.r_dirTheta >= self.config.eval.cuts.theta) or
               (event.r_dirPhi <= -self.config.eval.cuts.phi) or
               (event.r_dirPhi >= self.config.eval.cuts.phi))
        return cut

    def cosmic_cut(self, event):
        """Calculate if the event should be cut due to the cosmic network output.
        Args:
            event (dict): Pandas event(row) dict
        Returns:
            bool: cut or not?
        """
        return (event.c_out >= self.config.eval.cuts.cosmic)

    def classify(self, event):
        """Add the correct weight to each event.
        Args:
            event (dict): Pandas event(row) dict
        Returns:
            int: category classification
        """
        if event['cosmic_cut']:
            return self.beam_model.categories
        else:
            x = [event[('b_out_' + str(i))] for i in range(self.beam_model.categories)]
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

    def save_to_file(self):
        """Save everything to a ROOT file.
        """
        print('--- saving to file... ', end='', flush=True)
        # to_root(self.events, self.config.eval.output, key='events')
        print('done')
