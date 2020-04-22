# -*- coding: utf-8 -*-

"""
Training classes for fitting all the models

This module contains all the training classes used to fit the different
models. All are derived from the BaseTrainer class and allow for
customisation of the training loop if required.
"""

import os
import csv
from tensorflow.keras import callbacks


class BaseTrainer(object):

    """
    Base model class which all implementations derive from.
    """

    def __init__(self, config, model, data):
        """
        Initialise the BaseTrainer.

        Args:
            config (dotmap.DotMap): DotMap Configuration namespace
            model (chipscvn.models model): Model to use with trainer
            data (chipscvn.data.DataLoader): Data loader
        """
        self.config = config
        self.model = model
        self.data = data
        self.callbacks = []
        self.history = []

    def init_callbacks(self):
        """
        Initialise the keras callbacks, overide in derived model class.
        """
        raise NotImplementedError

    def train(self):
        """
        Train the model, overide in derived model class.
        """
        raise NotImplementedError


class BasicTrainer(BaseTrainer):

    """
    Basic trainer class, implements a simple keras fitter.
    """

    def __init__(self, config, model, data):
        """
        Initialise the BasicTrainer.

        Args:
            config (dotmap.DotMap): DotMap Configuration namespace
            model (chipscvn.models model): Model to use with trainer
            data (chipscvn.data.DataLoader): Data loader
        """
        super().__init__(config, model, data)
        self.init_callbacks()

    def lr_scheduler(self, epoch):
        """
        Learning rate schedule function.

        Args:
            epoch (int): Epoch number
        Returns:
            float: The generated dataset
        """
        lr = self.config.model.lr * 1/(1 + self.config.model.lr_decay * epoch)
        print("\nLearning Rate: {}".format(lr))
        return lr

    def init_callbacks(self):
        """
        Initialise the keras callbacks to be called at the end of each epoch.
        """
        self.callbacks.append(  # Tensorboard callback for viewing plots of model training
            callbacks.TensorBoard(
                log_dir=self.config.exp.tensorboard_dir,
                histogram_freq=1,
            )
        )

        self.callbacks.append(  # Checkpoint callback to save model weights at epoch end
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config.exp.checkpoints_dir, 'cp-{epoch:04d}.ckpt'),
                save_weights_only=True,
                verbose=0)
        )

        self.callbacks.append(  # Early stopping callback to stop training if not improving
            callbacks.EarlyStopping(
                monitor=self.model.es_monitor,
                min_delta=self.config.trainer.es_delta,
                patience=self.config.trainer.es_epochs,
                verbose=1,
                mode='auto')
        )

        # Learning rate scheduler callback
        self.callbacks.append(callbacks.LearningRateScheduler(self.lr_scheduler))

    def train(self, additional_callbacks=[]):
        """
        Train the model using the tf.keras fit api call.
        """
        self.callbacks.extend(additional_callbacks)

        if self.config.trainer.steps_per_epoch == -1:
            steps = None
        else:
            steps = self.config.trainer.steps_per_epoch

        self.history = self.model.model.fit(
            self.data.train_data(),
            epochs=self.config.trainer.num_epochs,
            verbose=1,
            validation_data=self.data.val_data(),
            callbacks=self.callbacks,
            steps_per_epoch=steps
        )

    def save(self):
        """
        Saves the training history to file.
        """
        history_name = os.path.join(self.config.exp.exp_dir, 'history.csv')
        with open(history_name, mode='w') as history_file:
            writer = csv.DictWriter(history_file, self.history.history.keys())
            writer.writeheader()
            writer.writerow(self.history.history)
