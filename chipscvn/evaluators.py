"""Training classes for fitting all the models

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module contains all the training classes used to fit the different
models. All are derived from the BaseTrainer class and allow for
customisation of the training loop if required.
"""

import time

import pandas as pd
import numpy as np
from tensorflow.keras import Model
from sklearn.preprocessing import StandardScaler

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


class ClassificationEvaluator(BaseEvaluator):
    """Implements the classification evaluation."""
    def __init__(self, config):
        super().__init__(config)

    def init_evaluator(self):
        """Initialise the classification evaluator, loading the evaluation data and model."""
        self.config.data.input_dirs = self.config.eval.input_dirs

        # Get the model to evaluate and load the most recent weights
        self.model = utils.get_model(self.config)
        self.model.load()

        # Load the test dataset into memory for easier manipulation
        data_loader = data.DataLoader(self.config)
        self.data = data_loader.test_data()

    def run(self):
        """Run the evaluator, just running a simple test on the model."""
        print('--- Running Evaluation ---\n')
        start_time = time.time()  # Time how long it takes
        events = {  # Dict to hold all the event data
            'true_pdg': [],
            'true_type': [],
            'true_category': [],
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
            'predictions': [],
            'dense_output': [],
            'ct_image': [],
            'h_image': []
        }

        # Create a new model that outputs the dense layer of the original model
        dense_model = Model(inputs=self.model.model.input,
                            outputs=self.model.model.get_layer('dense_final').output)

        for x, y in self.data:  # Run prediction on individual batches
            # Fill events dict with 'inputs' data
            for name, array in list(x.items()):
                if name in events.keys():
                    events[name].extend(array.numpy())

            # Fill events dict with 'labels' data
            for name, array in list(y.items()):
                if name in events.keys():
                    events[name].extend(array.numpy())

            events['predictions'].extend(self.model.model.predict(x))
            events['dense_output'].extend(dense_model.predict(x))

        self.events = pd.DataFrame.from_dict(events)
        print('--- %s seconds to test model ---' % (time.time() - start_time))


class EnergyEvaluator(BaseEvaluator):
    """Implements the energy estimation evaluation."""
    def __init__(self, config):
        super().__init__(config)

    def init_evaluator(self):
        """Initialise the energy estimation evaluator, loading the evaluation data and model."""
        self.config.data.input_dirs = self.config.eval.input_dirs

        # Get the model to evaluate and load the most recent weights
        self.model = utils.get_model(self.config)
        self.model.load()

        # Load the test dataset into memory for easier manipulation
        data_loader = data.DataLoader(self.config)
        self.data = data_loader.test_data()

    def run(self):
        """Run the evaluator, just running a simple test on the model."""
        print('--- Running Evaluation ---\n')
        start_time = time.time()  # Time how long it takes
        events = {  # Dict to hold all the event data
            'true_pdg': [],
            'true_type': [],
            'true_category': [],
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
            'pred_nuEnergy': [],
            'ct_image': [],
            'h_image': []
        }

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

        self.events = pd.DataFrame.from_dict(events)
        print('--- %s seconds to test model ---' % (time.time() - start_time))

