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
        self.predict()
        self.predict_dense()

    def predict(self):
        """Make predictions on some data given a model."""
        start_time = time.time()  # Time how long it takes
        pred_dict = {parameter: [] for parameter in self.model.parameters}
        labels_dict = {
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

        for x, y in self.data:  # Run prediction on individual batches
            predictions = self.model.model.predict(x)
            for counter, key in enumerate(pred_dict.keys()):
                if len(pred_dict) == 1:
                    pred_dict[key].extend(predictions)
                else:
                    pred_dict[key].extend(predictions[counter])
            for name, array in list(y.items()):
                labels_dict[name].append(array.numpy())

        for key in labels_dict:
            labels_dict[key] = np.concatenate(labels_dict[key]).ravel()

        self.labels = pd.DataFrame.from_dict(labels_dict)
        self.predictions = pd.DataFrame.from_dict(pred_dict)
        print("--- %s seconds to test model ---" % (time.time() - start_time))

    def predict_dense(self):
        """Get dense layer output on some data given a model."""
        start_time = time.time()  # Time how long it takes

        # Create a new model that outputs the dense layer of the original model
        dense_model = Model(inputs=self.model.model.input,
                            outputs=self.model.model.get_layer('dense').output)

        dense_list = []
        labels_dict = {
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

        for x, y in self.data:  # Run prediction on individual batches
            dense_list.append(dense_model.predict(x))
            for name, array in list(y.items()):
                labels_dict[name].append(array.numpy())

        for key in labels_dict:
            labels_dict[key] = np.concatenate(labels_dict[key]).ravel()

        self.dense_labels = pd.DataFrame.from_dict(labels_dict)
        self.dense_output = pd.DataFrame(np.concatenate(dense_list))
        self.dense_output = StandardScaler().fit_transform(dense_output)
        print("--- %s seconds to test model ---" % (time.time() - start_time))