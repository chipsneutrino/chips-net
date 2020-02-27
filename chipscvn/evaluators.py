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


class BasicEvaluator(BaseEvaluator):
    """Implements a basic model evaluation."""
    def __init__(self, config):
        super().__init__(config)

    def init_evaluator(self):
        """Initialise the basic evaluator, loading the evaluation data and model."""
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
        inputs_dict = {  # Dict to hold all inputs
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
            'z_cut': [],
            'r_cut': [],
            'dir_cut': [],
            'q_cut': []
        }
        labels_dict = {  # Dict to hold all labels
            'true_pdg': [],
            'true_type': [],
            'true_category': [],
            'true_vtxX': [],
            'true_vtxY': [],
            'true_vtxZ': [],
            'true_dirTheta': [],
            'true_dirPhi': [],
            'true_nuEnergy': [],
            'true_lepEnergy': []
        }
        pred_dict = {  # Dict to hold predicted parameters from the model
            parameter: [] for parameter in self.model.parameters
        }
        dense_list = []  # List to hold final dense layer activations

        # Create a new model that outputs the dense layer of the original model
        dense_model = Model(inputs=self.model.model.input,
                            outputs=self.model.model.get_layer('dense_final').output)

        for x, y in self.data:  # Run prediction on individual batches
            # Fill inputs dict
            for name, array in list(x.items()):
                if name in inputs_dict.keys():
                    inputs_dict[name].append(array.numpy())

            # Fill label dict
            for name, array in list(y.items()):
                labels_dict[name].append(array.numpy())

            # Fill predictions dict
            predictions = self.model.model.predict(x)
            for counter, key in enumerate(pred_dict.keys()):
                if len(pred_dict) == 1:
                    pred_dict[key].extend(predictions)
                else:
                    pred_dict[key].extend(predictions[counter])

            # Fill dense layer activations list
            dense_list.append(dense_model.predict(x))

        for key in inputs_dict:
            inputs_dict[key] = np.concatenate(inputs_dict[key]).ravel()

        for key in labels_dict:
            labels_dict[key] = np.concatenate(labels_dict[key]).ravel()

        self.inputs = pd.DataFrame.from_dict(inputs_dict)
        self.labels = pd.DataFrame.from_dict(labels_dict)
        self.predictions = pd.DataFrame.from_dict(pred_dict)
        self.dense = pd.DataFrame(np.concatenate(dense_list))
        self.dense = StandardScaler().fit_transform(self.dense)
        print('--- %s seconds to test model ---' % (time.time() - start_time))
