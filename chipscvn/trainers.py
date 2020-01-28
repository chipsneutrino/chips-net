"""Training classes for fitting all the models

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module contains all the training classes used to fit the different
models. All are derived from the BaseTrainer class and allow for
customisation of the training loop if required.
"""

import os
from tensorflow.keras import callbacks


class BaseTrainer(object):
    """Base model class which all implementations derive from."""
    def __init__(self, config, model, data):
        self.config = config
        self.model = model
        self.data = data
        self.callbacks = []
        self.history = []

    def init_callbacks(self):
        """Initialise the keras callbacks, overide in derived model class."""
        raise NotImplementedError

    def train(self):
        """Train the model, overide in derived model class."""
        raise NotImplementedError


class BasicTrainer(BaseTrainer):
    """Build the model, overide in derived model class."""
    def __init__(self, config, model, data):
        super().__init__(config, model, data)
        self.init_callbacks()

    def init_callbacks(self):
        """Initialise the keras callbacks to be called at the end of each epoch."""
        self.callbacks.append(  # Tensorboard callback for viewing plots of model training
            callbacks.TensorBoard(
                log_dir=self.config.exp_dir,
                histogram_freq=1,
            )
        )

        self.callbacks.append(  # Checkpoint callback to save model weights at epoch end
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config.exp_dir, "cp-{epoch:04d}.ckpt"),
                save_weights_only=True,
                verbose=0)
        )

        self.callbacks.append(  # Early stopping callback to stop training if not improving
            callbacks.EarlyStopping(
                monitor=self.model.es_monitor,
                min_delta=self.config.es_delta,
                patience=self.config.es_epochs,
                verbose=1,
                mode='min')
        )

    def train(self):
        """Train the model using the keras fit api call."""
        train_ds = self.data.train_data().take(self.config.max_examples)
        val_ds = self.data.val_data().take(250)

        self.history = self.model.model.fit(
            train_ds,
            epochs=self.config.train_epochs,
            verbose=1,
            validation_data=val_ds,
            callbacks=self.callbacks,
        )
