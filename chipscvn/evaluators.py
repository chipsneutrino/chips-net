"""Training classes for fitting all the models

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module contains all the training classes used to fit the different
models. All are derived from the BaseTrainer class and allow for
customisation of the training loop if required.
"""

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
    """Basic evaluator class, implements a simple keras API fitter."""
    def __init__(self, config):
        super().__init__(config)

    def init_evaluator(self):
        """Initialise the basic evaluator, loading the evaluation data and model."""
        self.config.data.input_dirs = self.config.eval.input_dirs
        self.data = data.DataLoader(self.config)
        self.model = utils.get_model(self.config)

    def run(self):
        """Run the evaluator, just running a simple test on the model."""
        self.model.load()
        result = self.model.model.evaluate(self.data.test_data())
        print(result)
