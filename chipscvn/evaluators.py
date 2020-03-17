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


class CombinedEvaluator(BaseEvaluator):
    """Combined Cosmic and Beam classification model evaluation."""
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

        self.run_inference()  # Get model outputs for test data
        self.parse_outputs()  # Parse the outputs into other quantities
        self.calculate_weights()  # Calculate the weights to apply categorically
        self.calculate_cuts()  # Calculate different cuts to be applied

        print('--- Done (took %s seconds) ---\n' % (time.time() - start_time))

    def run_inference(self):
        """Get model outputs for test data."""
        print('--- running inference...\n')

        events = {  # Create empty dict to hold all the event data
            'true_pdg': [], 'true_type': [], 'true_category': [], 'true_cosmic': [],
            'true_vtxX': [], 'true_vtxY': [], 'true_vtxZ': [], 'true_dirTheta': [],
            'true_dirPhi': [], 'true_nuEnergy': [], 'true_lepEnergy': [], 'raw_num_hits': [],
            'filtered_num_hits': [], 'num_hough_rings': [], 'raw_total_digi_q': [],
            'filtered_total_digi_q': [], 'first_ring_height': [], 'last_ring_height': [],
            'reco_vtxX': [], 'reco_vtxY': [], 'reco_vtxZ': [], 'reco_dirTheta': [],
            'reco_dirPhi': [], 'cosmic_output': [], 'beam_output': [], 'cosmic_dense': [],
            'beam_dense': []
        }

        images = self.config.data.img_size[2]
        if self.config.data.stack:
            images = 1
        for i in range(images):
            events[('image_' + str(i))] = []

        cosmic_dense_model = Model(  # Model to ouput cosmic model dense layer
            inputs=self.cosmic_model.model.input,
            outputs=self.cosmic_model.model.get_layer('dense_final').output
        )

        beam_dense_model = Model(  # Model to ouput beam model dense layer
            inputs=self.beam_model.model.input,
            outputs=self.beam_model.model.get_layer('dense_final').output
        )

        for x, y in self.data:  # Run prediction on individual batches
            for name, array in list(x.items()):  # Fill events dict with 'inputs'
                if name in events.keys():
                    events[name].extend(array.numpy())

            for name, array in list(y.items()):  # Fill events dict with 'labels'
                if name in events.keys():
                    events[name].extend(array.numpy())

            events['cosmic_output'].extend(self.cosmic_model.model.predict(x))
            events['beam_output'].extend(self.beam_model.model.predict(x))
            events['cosmic_dense'].extend(cosmic_dense_model.predict(x))
            events['beam_dense'].extend(beam_dense_model.predict(x))

        self.events = pd.DataFrame.from_dict(events)  # Convert dict to pandas dataframe

    def parse_outputs(self):
        """Parse the outputs into other quantities."""
        print('--- parsing outputs...\n')

        # Calculate the true combined category for each event
        self.events['true_cat_combined'] = self.events.apply(self.true_classifier, axis=1)

        # Parse outputs into easier to use pandas columns including combined outputs
        self.events['cosmic_output'] = self.events.cosmic_output.map(lambda x: x[0])
        for i in range(9):
            self.events[('beam_output_' + str(i))] = self.events.beam_output.map(lambda x: x[i])

        self.events['nuel_cc_combined'] = self.events.apply(self.nuel_cc_combined, axis=1)
        self.events['numu_cc_combined'] = self.events.apply(self.numu_cc_combined, axis=1)
        self.events.drop('beam_output', axis=1, inplace=True)

    def true_classifier(self, event):
        """Classify events into one of the 4 true categories."""
        if event['true_category'] in [0, 2, 4, 6]:
            return int(0)
        elif event['true_category'] in [1, 3, 5, 7]:
            return int(1)
        elif event['true_category'] == 8:
            return int(2)
        elif event['true_category'] == 9:
            return int(3)
        else:
            raise NotImplementedError

    def nuel_cc_combined(self, event):
        """Combine the 4 nuel CC event types into a single category score."""
        return (event['beam_output_0'] +
                event['beam_output_2'] +
                event['beam_output_4'] +
                event['beam_output_6'])

    def numu_cc_combined(self, event):
        """Combine the 4 numu CC event types into a single category score."""
        return (event['beam_output_1'] +
                event['beam_output_3'] +
                event['beam_output_5'] +
                event['beam_output_7'])

    def calculate_weights(self):
        """Calculate the weights to apply categorically."""
        print('--- calculating weights...\n')

        tot_nuel = self.events[(self.events.true_pdg == 0) & (self.events.true_cosmic == 0)].shape[0]
        tot_numu = self.events[(self.events.true_pdg == 1) & (self.events.true_cosmic == 0)].shape[0]
        tot_cosmic = self.events[self.events.true_cosmic == 1].shape[0]
        print("Total-> Nuel: {}, Numu: {}, Cosmic: {}\n".format(tot_nuel, tot_numu, tot_cosmic))

        self.nuel_weight = (1.0/tot_nuel)*(self.config.eval.weights.nuel*self.config.eval.weights.total)
        self.numu_weight = (1.0/tot_numu)*(self.config.eval.weights.numu*self.config.eval.weights.total)
        self.cosmic_weight = (1.0/tot_cosmic)*(self.config.eval.weights.cosmic*self.config.eval.weights.total)

        print('Weights-> Nuel:{0:.4f}, Numu:{1:.4f}, Cosmic:{2:.4f}\n'.format(
            self.nuel_weight, self.numu_weight, self.cosmic_weight)
        )

        self.events['weight'] = self.events.apply(self.add_weight, axis=1)

    def add_weight(self, event):
        """Add the correct weight to each event."""
        if event['true_pdg'] == 0 and event['true_cosmic'] == 0:
            return self.nuel_weight
        elif event['true_pdg'] == 1 and event['true_cosmic'] == 0:
            return self.numu_weight
        elif event['true_cosmic'] == 1:
            return self.cosmic_weight
        else:
            raise NotImplementedError

    def calculate_cuts(self):
        """Calculate different cuts to be applied."""
        print('--- calculating cuts...\n')

        self.events['cosmic_cut'] = self.events.apply(self.cosmic_cut, axis=1)
        self.events['activity_cut'] = self.events.apply(self.activity_cut, axis=1)
        self.events['dir_cut'] = self.events.apply(self.dir_cut, axis=1)
        self.events['cut'] = self.events.apply(self.combine_cuts, axis=1)

        for i in range(4):
            cat_events = self.events[self.events.true_cat_combined == i]
            print("Cat {}-> Total {}, Survived: {}\n".format(
                i, cat_events.shape[0],
                cat_events[cat_events['cut'] == False].shape[0]/cat_events.shape[0])
            )

    def cosmic_cut(self, event):
        """Calculate if the event should be cut due to the cosmic network output."""
        return (event['cosmic_output'] >= self.config.eval.cuts.cosmic)

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
        return event['cosmic_cut'] or event['activity_cut'] or event['dir_cut']

    def make_cat_plot(self, parameter, bins, low, high, scale='norm', cut=True):
        """Make the histograms and legend for a parameter, data is split in categories."""

        hists = []
        colours = [ROOT.kGreen, ROOT.kBlue, ROOT.kRed, ROOT.kBlack]
        leg = ROOT.TLegend(0.65, 0.65, 0.89, 0.89, "Event Type")
        entries = ["#nu_{e} CC", "#nu_{#mu} CC", "NC", "Cosmic"]
        for i in range(4):
            name = "h_" + str(i)
            hist = ROOT.TH1F(name, parameter, bins, low, high)
            hist.SetLineColor(colours[i])
            hist.SetLineWidth(2)
            hist.GetXaxis().SetTitle(parameter)
            events = self.events[self.events['true_cat_combined'] == i]
            if cut and scale == 'weight':
                fill_hist(hist,
                          events.loc[events['cut'] == False][parameter].values,
                          events.loc[events['cut'] == False]['weight'].values)
            elif not cut and scale == 'weight':
                fill_hist(hist, events[parameter].values, events['weight'].values)
            elif cut and scale == 'norm':
                fill_hist(hist, events.loc[events['cut'] == False][parameter].values)
                hist.Scale(1.0/hist.GetEntries())
            elif not cut and scale == 'norm':
                fill_hist(hist, events[parameter].values)
                hist.Scale(1.0/hist.GetEntries())
            else:
                raise NotImplementedError
            hists.append(hist)
            leg.AddEntry(hists[i], entries[i], "PL")

        leg.SetTextSize(0.03)
        leg.SetTextFont(42)
        leg.SetBorderSize(0)

        return hists, leg
