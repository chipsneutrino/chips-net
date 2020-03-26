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
        c_config = self.config
        c_config.model = self.config.models.cosmic
        c_config.exp.experiment_dir = self.config.models.cosmic.dir
        c_config.exp.name = self.config.models.cosmic.path
        chipscvn.config.setup_dirs(c_config, False)
        self.c_model = utils.get_model(c_config)
        self.c_model.load()

        # Get the beam classification model and load the most recent weights
        b_config = self.config
        b_config.model = self.config.models.beam
        b_config.exp.experiment_dir = self.config.models.beam.dir
        b_config.exp.name = self.config.models.beam.path
        chipscvn.config.setup_dirs(b_config, False)
        self.b_model = utils.get_model(b_config)
        self.b_model.load()

        # Get the test dataset
        data_loader = data.DataLoader(self.config)
        self.data = data_loader.test_data()

        # Fully combined category names
        self.comb_cat_names = ['Nuel-CC', 'Numu-CC', 'NC', 'Cosmic']

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
            't_nu': [], 't_code': [], 't_cat': [], 't_cosmic_cat': [],
            't_full_cat': [], 't_nu_nc_cat': [], 't_nc_cat': [],
            't_vtxX': [], 't_vtxY': [], 't_vtxZ': [], 't_vtxT': [], 't_nuEnergy': [],
            't_p_pdgs': [], 't_p_energies': [], 't_p_dirTheta': [], 't_p_dirPhi': [],
            'r_raw_num_hits': [], 'r_filtered_num_hits': [], 'r_num_hough_rings': [],
            'r_raw_total_digi_q': [], 'r_filtered_total_digi_q': [],
            'r_first_ring_height': [], 'r_last_ring_height': [],
            'r_vtxX': [], 'r_vtxY': [], 'r_vtxZ': [], 'r_vtxT': [],
            'r_dirTheta': [], 'r_dirPhi': [],
            'c_out': [], 'b_out': [], 'c_dense': [], 'b_dense': []
        }

        images = self.config.data.img_size[2]
        if self.config.data.stack:
            images = 1
        for i in range(images):
            events[('image_' + str(i))] = []

        c_dense_model = Model(  # Model to ouput cosmic model dense layer
            inputs=self.c_model.model.input,
            outputs=self.c_model.model.get_layer('dense_final').output
        )

        b_dense_model = Model(  # Model to ouput beam model dense layer
            inputs=self.b_model.model.input,
            outputs=self.b_model.model.get_layer('dense_final').output
        )

        for x, y in self.data:  # Run prediction on individual batches
            for name, array in list(x.items()):  # Fill events dict with 'inputs'
                if name in events.keys():
                    events[name].extend(array.numpy())

            for name, array in list(y.items()):  # Fill events dict with 'labels'
                if name in events.keys():
                    events[name].extend(array.numpy())

            events['c_out'].extend(self.c_model.model.predict(x))
            events['b_out'].extend(self.b_model.model.predict(x))
            events['c_dense'].extend(c_dense_model.predict(x))
            events['b_dense'].extend(b_dense_model.predict(x))

        self.events = pd.DataFrame.from_dict(events)  # Convert dict to pandas dataframe
        self.events = self.events.sample(frac=1).reset_index(drop=True)  # Shuffle the dataframe

    def parse_outputs(self):
        """Parse the outputs into other quantities."""
        print('--- parsing outputs...\n')

        # Parse outputs into easier to use pandas columns including combined outputs
        self.events.c_out = self.events.c_out.map(lambda x: x[0])
        for i in range(self.b_model.categories):
            self.events[('b_out_' + str(i))] = self.events.b_out.map(lambda x: x[i])

        self.events['scores'] = self.events.apply(self.b_model.combine_outputs, axis=1)
        self.events['nuel_score'] = self.events.scores.map(lambda x: x[0])
        self.events['numu_score'] = self.events.scores.map(lambda x: x[1])
        self.events['nc_score'] = self.events.scores.map(lambda x: x[2])
        self.events.drop('b_out', axis=1, inplace=True)
        self.events.drop('scores', axis=1, inplace=True)

    def calculate_weights(self):
        """Calculate the weights to apply categorically."""
        print('--- calculating weights...\n')

        tot_nuel = self.events[(self.events.t_nu == 0) &
                               (self.events.t_cosmic_cat == 0)].shape[0]
        tot_numu = self.events[(self.events.t_nu == 1) &
                               (self.events.t_cosmic_cat == 0)].shape[0]
        tot_cosmic = self.events[self.events.t_cosmic_cat == 1].shape[0]
        print("Total-> Nuel: {}, Numu: {}, Cosmic: {}\n".format(tot_nuel, tot_numu, tot_cosmic))

        self.nuel_weight = (1.0/tot_nuel)*(self.config.eval.weights.nuel *
                                           self.config.eval.weights.total)
        self.numu_weight = (1.0/tot_numu)*(self.config.eval.weights.numu *
                                           self.config.eval.weights.total)
        self.cosmic_weight = (1.0/tot_cosmic)*(self.config.eval.weights.cosmic *
                                               self.config.eval.weights.total)

        print('Weights-> Nuel:{0:.4f}, Numu:{1:.4f}, Cosmic:{2:.4f}\n'.format(
            self.nuel_weight, self.numu_weight, self.cosmic_weight)
        )

        self.events['weight'] = self.events.apply(self.add_weight, axis=1)

    def add_weight(self, event):
        """Add the correct weight to each event."""
        if event.t_nu == 0 and event.t_cosmic_cat == 0:
            return self.nuel_weight
        elif event.t_nu == 1 and event.t_cosmic_cat == 0:
            return self.numu_weight
        elif event.t_cosmic_cat == 1:
            return self.cosmic_weight
        else:
            raise NotImplementedError

    def calculate_cuts(self):
        """Calculate different cuts to be applied."""
        print('--- calculating cuts...\n')

        self.events['base_cut'] = self.events.apply(self.base_cut, axis=1)
        self.events['cosmic_cut'] = self.events.apply(self.cosmic_cut, axis=1)

    def base_cut_summary(self):
        """Print how each category is affected by the base_cut."""
        print("Base Cut Summary...\n")
        for i in range(4):
            cat_events = self.events[self.events.t_full_cat == i]
            print("{}-> Total {}, Survived: {}\n".format(
                self.comb_cat_names[i], cat_events.shape[0],
                cat_events[cat_events['base_cut'] == 0].shape[0]/cat_events.shape[0]))

    def combined_cut_summary(self):
        """Print how each category is affected by the base_cut + cosmic_cut."""
        print("Base + Cosmic Cut Summary...\n")
        for i in range(4):
            cat_events = self.events[self.events.t_full_cat == i]
            print("{}-> Total {}, Survived: {}\n".format(
                self.comb_cat_names[i], cat_events.shape[0], cat_events[
                    (cat_events.base_cut == 0) &
                    (cat_events.cosmic_cut == 0)].shape[0]/cat_events.shape[0]))

    def base_cut(self, event):
        """Calculate if the event should be cut due to activity."""
        cut = ((event.r_raw_total_digi_q <= self.config.eval.cuts.q) or
               (event.r_first_ring_height <= self.config.eval.cuts.hough) or
               (event.r_dirTheta <= -self.config.eval.cuts.theta) or
               (event.r_dirTheta >= self.config.eval.cuts.theta) or
               (event.r_dirPhi <= -self.config.eval.cuts.phi) or
               (event.r_dirPhi >= self.config.eval.cuts.phi))
        return cut

    def cosmic_cut(self, event):
        """Calculate if the event should be cut due to the cosmic network output."""
        return (event.c_out >= self.config.eval.cuts.cosmic)

    def cat_plot(self, parameter, bins, x_low, x_high, y_low, y_high, scale='norm',
                 base_cut=True, cosmic_cut=True):
        """Make the histograms and legend for a parameter, split by true category."""

        # 0 = Nuel CC-QEL
        # 1 = Nuel CC-RES
        # 2 = Nuel CC-DIS
        # 3 = Nuel CC-COH
        # 4 = Numu CC-QEL
        # 5 = Numu CC-RES
        # 6 = Numu CC-DIS
        # 7 = Numu CC-COH
        # 8 = Nuel NC-QEL
        # 9 = Nuel NC-RES
        # 10 = Nuel NC-DIS
        # 11 = Nuel NC-COH
        # 12 = Numu NC-QEL
        # 13 = Numu NC-RES
        # 14 = Numu NC-DIS
        # 15 = Numu NC-COH
        # 16 = Cosmic

        hists = []
        colours = [ROOT.kGreen, ROOT.kGreen+1, ROOT.kGreen+2, ROOT.kGreen+3,
                   ROOT.kBlue, ROOT.kBlue+1, ROOT.kBlue+2, ROOT.kBlue+3,
                   ROOT.kRed, ROOT.kRed+1, ROOT.kRed+2, ROOT.kRed+3,
                   ROOT.kOrange, ROOT.kOrange+1, ROOT.kOrange+2, ROOT.kOrange+3,
                   ROOT.kBlack]
        leg = ROOT.TLegend(0.65, 0.65, 0.80, 0.89, "Event Type")
        entries = ["EL-CC-QEL", "EL-CC-RES", "EL-CC-DIS", "EL-CC-COH",
                   "MU-CC-QEL", "MU-CC-RES", "MU-CC-DIS", "MU-CC-COH",
                   "EL-NC-QEL", "EL-NC-RES", "EL-NC-DIS", "EL-NC-COH",
                   "MU-NC-QEL", "MU-NC-RES", "MU-NC-DIS", "MU-NC-COH",
                   "Cosmic"]
        for i in range(17):
            name = "h_" + parameter + "_" + entries[i]
            hist = ROOT.TH1F(name, parameter, bins, x_low, x_high)
            hist.SetLineColor(colours[i])
            hist.SetLineWidth(2)
            hist.GetXaxis().SetTitle(parameter)

            events = self.events[self.events['t_cat'] == i]

            if base_cut:
                events = events[events.base_cut == 0]
            if cosmic_cut:
                events = events[events.cosmic_cut == 0]

            if scale == 'weight':
                fill_hist(hist, events[parameter].to_numpy(), events['weight'].to_numpy())
            elif scale == 'norm':
                fill_hist(hist, events[parameter].to_numpy())
                if hist.GetEntries() > 0:
                    hist.Scale(1.0/hist.GetEntries())
            elif scale == 'none':
                fill_hist(hist, events[parameter].to_numpy())
            else:
                raise NotImplementedError
            hists.append(hist)
            leg.AddEntry(hists[i], entries[i], "PL")

        hists[0].GetYaxis().SetRangeUser(y_low, y_high)

        leg.SetTextSize(0.03)
        leg.SetTextFont(42)
        leg.SetBorderSize(0)

        return hists, leg

    def combined_cat_plot(self, parameter, bins, x_low, x_high, y_low, y_high, scale='norm',
                          base_cut=True, cosmic_cut=True):
        """Make the histograms and legend for a parameter, split by combined category."""

        hists = []
        colours = [ROOT.kGreen, ROOT.kBlue, ROOT.kRed, ROOT.kBlack]
        leg = ROOT.TLegend(0.65, 0.65, 0.80, 0.89, "Event Type")
        entries = ["EL-CC", "MU-CC", "NC", "Cosmic"]
        for i in range(4):
            name = "h_" + parameter + "_" + entries[i]
            hist = ROOT.TH1F(name, parameter, bins, x_low, x_high)
            hist.SetLineColor(colours[i])
            hist.SetLineWidth(2)
            hist.GetXaxis().SetTitle(parameter)

            events = self.events[self.events['t_full_cat'] == i]

            if base_cut:
                events = events[events.base_cut == 0]
            if cosmic_cut:
                events = events[events.cosmic_cut == 0]

            if scale == 'weight':
                fill_hist(hist, events[parameter].to_numpy(), events['weight'].to_numpy())
            elif scale == 'norm':
                fill_hist(hist, events[parameter].to_numpy())
                if hist.GetEntries() > 0:
                    hist.Scale(1.0/hist.GetEntries())
            elif scale == 'none':
                fill_hist(hist, events[parameter].to_numpy())
            else:
                raise NotImplementedError
            hists.append(hist)
            leg.AddEntry(hists[i], entries[i], "PL")

        hists[0].GetYaxis().SetRangeUser(y_low, y_high)

        leg.SetTextSize(0.03)
        leg.SetTextFont(42)
        leg.SetBorderSize(0)

        return hists, leg
