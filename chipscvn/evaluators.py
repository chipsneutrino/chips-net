"""Evaluator classes for testing the trained models

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module contains all the evaluation classes used to test the
trained models on test data to determine their performance. All are
derived from the BaseEvaluator class.
"""

import time

import pandas as pd
from tensorflow.keras import Model
import ROOT
from root_numpy import fill_hist

import chipscvn.config
import chipscvn.data as data
import chipscvn.utils as utils


class BaseEvaluator(object):
    """Base evaluator class which all implementations derive from."""
    def __init__(self, config):
        self.config = config
        self.init_evaluator()

    def init_evaluator(self):
        """Initialise the evaluator, overide in derived model class."""
        raise NotImplementedError

    def run(self):
        """Run the evaluator, overide in derived model class."""
        raise NotImplementedError


class EnergyEvaluator(BaseEvaluator):
    """Implements the energy estimation evaluation."""
    def __init__(self, config):
        super().__init__(config)

    def init_evaluator(self):
        """Initialise the evaluator, loading the evaluation data and model."""
        chipscvn.config.setup_dirs(self.config, False)

        # Get the model to evaluate and load the most recent weights
        self.model = utils.get_model(self.config)
        self.model.load()

        # Load the test dataset into memory for easier manipulation
        data_loader = data.DataLoader(self.config)
        self.data = data_loader.test_data()

    def run(self):
        """Run the full evaluation."""
        print('--- Running Evaluation ---\n')
        start_time = time.time()  # Time how long it takes
        events = {  # Dict to hold all the event data
            'true_pdg': [],
            'true_type': [],
            'true_category': [],
            'true_cosmic': [],
            'true_vtxX': [],
            'true_vtxY': [],
            'true_vtxZ': [],
            'true_dirTheta': [],
            'true_dirPhi': [],
            'true_nuEnergy': [],
            'true_lepEnergy': [],
            'raw_num_hits': [],
            'filtered_num_hits': [],
            'num_hough_rings': [],
            'raw_total_digi_q': [],
            'filtered_total_digi_q': [],
            'first_ring_height': [],
            'last_ring_height': [],
            'reco_vtxX': [],
            'reco_vtxY': [],
            'reco_vtxZ': [],
            'reco_dirTheta': [],
            'reco_dirPhi': [],
            'pred_nuEnergy': []
        }

        for i in range(self.config.data.img_size[2]):
            input_name = 'image_' + str(i)
            events[input_name] = []

        print('--- running inference...\n')

        for x, y in self.data:  # Run prediction on individual batches
            # Fill events dict with 'inputs' data
            for name, array in list(x.items()):
                if name in events.keys():
                    events[name].extend(array.numpy())

            # Fill events dict with 'labels' data
            for name, array in list(y.items()):
                if name in events.keys():
                    events[name].extend(array.numpy())

            events['pred_nuEnergy'].extend(self.model.model.predict(x))

        self.events = pd.DataFrame.from_dict(events)  # Convert dict to pandas dataframe

        print('--- applying cuts...\n')

        # Calculate if event is to be cut
        self.events['activity_cut'] = self.events.apply(self.activity_cut, axis=1)
        self.events['dir_cut'] = self.events.apply(self.dir_cut, axis=1)
        self.events['cut'] = self.events.apply(self.combine_cuts, axis=1)
        # TODO: Implement a fiducial cut!

        print('--- Done (took %s seconds) ---' % (time.time() - start_time))

    def activity_cut(self, event):
        """Calculate if the event should be cut due to activity."""
        cut = ((event['raw_total_digi_q'] <= self.config.eval.cuts.q) or
               (event['first_ring_height'] <= self.config.eval.cuts.hough))
        return cut

    def dir_cut(self, event):
        """Calculate if the event should be cut due to reconstructed beam direction."""
        cut = ((event['reco_dirTheta'] <= -self.config.eval.cuts.theta) or
               (event['reco_dirTheta'] >= self.config.eval.cuts.theta) or
               (event['reco_dirPhi'] <= -self.config.eval.cuts.phi) or
               (event['reco_dirPhi'] >= self.config.eval.cuts.phi))
        return cut

    def combine_cuts(self, event):
        """Combine all singular cuts to see if the event should be cut."""
        return event['activity_cut'] or event['dir_cut']


class ClassificationEvaluator(BaseEvaluator):
    """Combines Cosmic and Beam classification model evaluation."""
    def __init__(self, config):
        super().__init__(config)

    def init_evaluator(self):
        """Initialise the evaluator, loading the evaluation data and model."""

        # Get the cosmic classification model and load the most recent weights
        cosmic_config = self.config
        cosmic_config.model = self.config.models.cosmic
        cosmic_config.exp.experiment_dir = self.config.models.cosmic.dir
        cosmic_config.exp.name = self.config.models.cosmic.path
        chipscvn.config.setup_dirs(cosmic_config, False)
        self.cosmic_model = utils.get_model(cosmic_config)
        self.cosmic_model.load()

        # Get the beam classification model and load the most recent weights
        beam_config = self.config
        beam_config.model = self.config.models.beam
        beam_config.exp.experiment_dir = self.config.models.beam.dir
        beam_config.exp.name = self.config.models.beam.path
        chipscvn.config.setup_dirs(beam_config, False)
        self.beam_model = utils.get_model(beam_config)
        self.beam_model.load()

        # Get the test dataset
        data_loader = data.DataLoader(self.config)
        self.data = data_loader.test_data()

    def run(self):
        """Run the full evaluation."""
        print('--- Running Evaluation ---\n')
        start_time = time.time()  # Time how long it takes
        events = {  # Dict to hold all the event data
            'true_pdg': [],
            'true_type': [],
            'true_category': [],
            'true_cosmic': [],
            'true_vtxX': [],
            'true_vtxY': [],
            'true_vtxZ': [],
            'true_dirTheta': [],
            'true_dirPhi': [],
            'true_nuEnergy': [],
            'true_lepEnergy': [],
            'raw_num_hits': [],
            'filtered_num_hits': [],
            'num_hough_rings': [],
            'raw_total_digi_q': [],
            'filtered_total_digi_q': [],
            'first_ring_height': [],
            'last_ring_height': [],
            'reco_vtxX': [],
            'reco_vtxY': [],
            'reco_vtxZ': [],
            'reco_dirTheta': [],
            'reco_dirPhi': [],
            'cosmic_output': [],
            'beam_output': [],
            'cosmic_dense': [],
            'beam_dense': [],
        }

        for i in range(self.config.data.img_size[2]):
            input_name = 'image_' + str(i)
            events[input_name] = []

        # Create a new model that outputs the dense layer of the cosmic model
        cosmic_dense_model = Model(
            inputs=self.cosmic_model.model.input,
            outputs=self.cosmic_model.model.get_layer('dense_final').output
        )

        # Create a new model that outputs the dense layer of the beam model
        beam_dense_model = Model(
            inputs=self.beam_model.model.input,
            outputs=self.beam_model.model.get_layer('dense_final').output
        )

        print('--- running inference...\n')

        for x, y in self.data:  # Run prediction on individual batches
            # Fill events dict with 'inputs' data
            for name, array in list(x.items()):
                if name in events.keys():
                    events[name].extend(array.numpy())

            # Fill events dict with 'labels' data
            for name, array in list(y.items()):
                if name in events.keys():
                    events[name].extend(array.numpy())

            events['cosmic_output'].extend(self.cosmic_model.model.predict(x))
            events['beam_output'].extend(self.beam_model.model.predict(x))
            events['cosmic_dense'].extend(cosmic_dense_model.predict(x))
            events['beam_dense'].extend(beam_dense_model.predict(x))

        self.events = pd.DataFrame.from_dict(events)  # Convert dict to pandas dataframe

        print('--- combining outputs...\n')

        # Determine the combined true category
        self.events['true_cat_combined'] = self.events.apply(self.true_classifier, axis=1)
        self.events['true_cat_combined'] = self.events['true_cat_combined'].astype('int')

        # Combine outputs into final category prediction values
        self.events['cosmic_output'] = self.events.apply(self.cosmic_pred, axis=1)
        self.events['nuel_cc_pred'] = self.events.apply(self.nuel_cc_pred, axis=1)
        self.events['numu_cc_pred'] = self.events.apply(self.numu_cc_pred, axis=1)
        self.events['nc_pred'] = self.events.apply(self.nc_pred, axis=1)

        print('--- applying cuts...\n')

        # Calculate if event is to be cut
        self.events['activity_cut'] = self.events.apply(self.activity_cut, axis=1)
        self.events['dir_cut'] = self.events.apply(self.dir_cut, axis=1)
        self.events['cut'] = self.events.apply(self.combine_cuts, axis=1)
        # TODO: Implement a fiducial cut!

        print('--- calculating weights...\n')

        # Seperate into category specific dataframes for ease of use later
        self.events_0 = self.events[(self.events.true_cat_combined == 0) &
                                    (self.events.true_type == 1)]
        self.events_1 = self.events[(self.events.true_cat_combined == 0) &
                                    (self.events.true_type != 1)]
        self.events_2 = self.events[(self.events.true_cat_combined == 1) &
                                    (self.events.true_type == 1)]
        self.events_3 = self.events[(self.events.true_cat_combined == 1) &
                                    (self.events.true_type != 1)]
        self.events_4 = self.events[self.events.true_cat_combined == 2]
        self.events_5 = self.events[self.events.true_cat_combined == 3]

        # Calculate the weight to give each category
        self.w_0 = (1.0/self.events[self.events.true_cat_combined == 0].shape[0])*(
            self.config.eval.weights.nuel*self.config.eval.weights.total)
        self.w_1 = (1.0/self.events[self.events.true_cat_combined == 1].shape[0])*(
            self.config.eval.weights.numu*self.config.eval.weights.total)
        self.w_2 = (1.0/self.events[self.events.true_cat_combined == 2].shape[0])*(
            self.config.eval.weights.nc*self.config.eval.weights.total)
        self.w_3 = (1.0/self.events[self.events.true_cat_combined == 3].shape[0])*(
            self.config.eval.weights.cosmic*self.config.eval.weights.total)
        print('--- Num-> Nuel:{}, Numu:{}, NC:{}, Cosmic:{}\n'.format(self.events_0.shape[0],
                                                                      self.events_1.shape[0],
                                                                      self.events_2.shape[0],
                                                                      self.events_3.shape[0]))
        print('--- Weights-> Nuel:{0:.4f}, Numu:{1:.4f}, NC:{2:.4f}, Cosmic:{3:.4f}\n'.format(
            self.w_0, self.w_1, self.w_2, self.w_3))
        print('--- Done (took %s seconds) ---\n' % (time.time() - start_time))

    def cut_summary(self):
        """Print a summary of how the cuts effect the different categories."""
        self.events_0.loc[self.events_0['cut'] == False].shape[0]
        print("Nuel CC QE: {}, Survived: {}\n".format(
            self.events_0.shape[0],
            self.events_0.loc[self.events_0['cut'] == False].shape[0]/self.events_0.shape[0]))
        print("Nuel CC nQE: {}, Survived: {}\n".format(
            self.events_1.shape[0],
            self.events_1.loc[self.events_1['cut'] == False].shape[0]/self.events_1.shape[0]))
        print("Numu CC QE: {}, Survived: {}\n".format(
            self.events_2.shape[0],
            self.events_2.loc[self.events_2['cut'] == False].shape[0]/self.events_2.shape[0]))
        print("Numu CC nQE: {}, Survived: {}\n".format(
            self.events_3.shape[0],
            self.events_3.loc[self.events_3['cut'] == False].shape[0]/self.events_3.shape[0]))
        print("NC: {}, Survived: {}\n".format(
            self.events_4.shape[0],
            self.events_4.loc[self.events_4['cut'] == False].shape[0]/self.events_4.shape[0]))
        print("Cosmic: {}, Survived: {}\n".format(
            self.events_5.shape[0],
            self.events_5.loc[self.events_5['cut'] == False].shape[0]/self.events_5.shape[0]))

    def true_classifier(self, event):
        """Classify events into one of the 4 true categories."""
        if event['true_category'] in [0, 2, 4, 6]:
            return 0
        elif event['true_category'] in [1, 3, 5, 7]:
            return 1
        elif event['true_category'] == 8:
            return 2
        elif event['true_category'] == 9:
            return 3

    def cosmic_pred(self, event):
        """Just return the network NC score."""
        return event['cosmic_output'][0]

    def nuel_cc_pred(self, event):
        """Combine the 4 nuel CC event types into a single category score."""
        return event['beam_output'][0]+event['beam_output'][2]+event['beam_output'][4]+event['beam_output'][6]

    def numu_cc_pred(self, event):
        """Combine the 4 numu CC event types into a single category score."""
        return event['beam_output'][1]+event['beam_output'][3]+event['beam_output'][5]+event['beam_output'][7]

    def nc_pred(self, event):
        """Just return the network NC score."""
        return event['beam_output'][8]

    def activity_cut(self, event):
        """Calculate if the event should be cut due to activity."""
        cut = ((event['raw_total_digi_q'] <= self.config.eval.cuts.q) or
               (event['first_ring_height'] <= self.config.eval.cuts.hough))
        return cut

    def dir_cut(self, event):
        """Calculate if the event should be cut due to reconstructed beam direction."""
        cut = ((event['reco_dirTheta'] <= -self.config.eval.cuts.theta) or
               (event['reco_dirTheta'] >= self.config.eval.cuts.theta) or
               (event['reco_dirPhi'] <= -self.config.eval.cuts.phi) or
               (event['reco_dirPhi'] >= self.config.eval.cuts.phi))
        return cut

    def combine_cuts(self, event):
        """Combine all singular cuts to see if the event should be cut."""
        return event['activity_cut'] or event['dir_cut']

    def make_cat_plot(self, parameter, bins, low, high, scale='none', cut=False):
        """Make the histograms and legend for a parameter, data is split in categories."""
        h_0 = ROOT.TH1F("h_0", parameter, bins, low, high)
        h_0.SetLineColor(79)
        h_0.SetLineWidth(2)
        h_0.GetXaxis().SetTitle(parameter)
        if cut:
            fill_hist(h_0, self.events_0.loc[self.events_0['cut'] == False][parameter].values)
        else:
            fill_hist(h_0, self.events_0[parameter].values)

        h_1 = ROOT.TH1F("h_1", parameter, bins, low, high)
        h_1.SetLineColor(209)
        h_1.SetLineWidth(2)
        if cut:
            fill_hist(h_1, self.events_1.loc[self.events_1['cut'] == False][parameter].values)
        else:
            fill_hist(h_1, self.events_1[parameter].values)

        h_2 = ROOT.TH1F("h_2", parameter, bins, low, high)
        h_2.SetLineColor(64)
        h_2.SetLineWidth(2)
        if cut:
            fill_hist(h_2, self.events_2.loc[self.events_2['cut'] == False][parameter].values)
        else:
            fill_hist(h_2, self.events_2[parameter].values)

        h_3 = ROOT.TH1F("h_3", parameter, bins, low, high)
        h_3.SetLineColor(214)
        h_3.SetLineWidth(2)
        if cut:
            fill_hist(h_3, self.events_3.loc[self.events_3['cut'] == False][parameter].values)
        else:
            fill_hist(h_3, self.events_3[parameter].values)

        h_4 = ROOT.TH1F("h_4", parameter, bins, low, high)
        h_4.SetLineColor(ROOT.kRed)
        h_4.SetLineWidth(2)
        if cut:
            fill_hist(h_4, self.events_4.loc[self.events_4['cut'] == False][parameter].values)
        else:
            fill_hist(h_4, self.events_4[parameter].values)

        h_5 = ROOT.TH1F("h_5", parameter, bins, low, high)
        h_5.SetLineColor(ROOT.kBlack)
        h_5.SetLineWidth(2)
        if cut:
            fill_hist(h_5, self.events_5.loc[self.events_5['cut'] == False][parameter].values)
        else:
            fill_hist(h_5, self.events_5[parameter].values)

        leg = ROOT.TLegend(0.65, 0.65, 0.89, 0.89, "Event Type")
        leg.AddEntry(h_0, "#nu_{e} CC QE", "PL")
        leg.AddEntry(h_1, "#nu_{e} CC nQE", "PL")
        leg.AddEntry(h_2, "#nu_{#mu} CC QE", "PL")
        leg.AddEntry(h_3, "#nu_{#mu} CC nQE", "PL")
        leg.AddEntry(h_4, "NC", "PL")
        leg.AddEntry(h_5, "Cosmic", "PL")
        leg.SetTextSize(0.03)
        leg.SetTextFont(42)
        leg.SetBorderSize(0)

        if scale == 'norm':
            h_0.GetYaxis().SetTitle("Fraction of Events")
            h_0.Scale(1.0/h_0.GetEntries())
            h_1.Scale(1.0/h_1.GetEntries())
            h_2.Scale(1.0/h_2.GetEntries())
            h_3.Scale(1.0/h_3.GetEntries())
            h_4.Scale(1.0/h_4.GetEntries())
            h_5.Scale(1.0/h_5.GetEntries())
        elif scale == 'weight':
            h_0.GetYaxis().SetTitle("Number of Events Per Year")
            h_0.Scale(self.w_0)
            h_1.Scale(self.w_0)
            h_2.Scale(self.w_1)
            h_3.Scale(self.w_1)
            h_4.Scale(self.w_2)
            h_5.Scale(self.w_3)
        elif scale == 'none':
            h_0.GetYaxis().SetTitle("Frequency")
        else:
            print("Error: Don't understand the scaling command")
            return

        return [h_0, h_1, h_2, h_3, h_4, h_5], leg
