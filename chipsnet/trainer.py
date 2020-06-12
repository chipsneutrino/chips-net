# -*- coding: utf-8 -*-

"""Module containing trainer class
"""

import os
import csv
import tensorflow as tf


class Trainer(object):
    """chipsnet trainer class, implements a keras fitter and associated utilities.
    """
    def __init__(self, config, model, data):
        """Initialise the BasicTrainer.
        Args:
            config (dotmap.DotMap): DotMap Configuration namespace
            model (chipsnet.models model): Model to use with trainer
            data (chipsnet.data.Reader): Data reader
        """
        self.config = config
        self.model = model
        self.data = data
        self.callbacks = []
        self.history = None
        self.init_callbacks()

    def lr_scheduler(self, epoch):
        """Learning rate schedule function.
        Args:
            epoch (int): Epoch number
        Returns:
            float: The generated dataset
        """
        lr = self.config.model.lr * 1/(1 + self.config.model.lr_decay * epoch)
        return lr

    def init_callbacks(self):
        """Initialise the keras callbacks to be called at the end of each epoch.
        """
        self.callbacks.append(  # Tensorboard callback for viewing plots of model training
            tf.keras.callbacks.TensorBoard(
                log_dir=self.config.exp.tensorboard_dir,
                histogram_freq=1,
                update_freq=self.config.trainer.tb_update,
                write_images=True,
                profile_batch='1,10'
            )
        )

        self.callbacks.append(  # Checkpoint callback to save model weights at epoch end
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config.exp.checkpoints_dir, 'cp-{epoch:04d}.ckpt'),
                save_weights_only=True,
                verbose=0)
        )

        self.callbacks.append(  # Early stopping callback to stop training if not improving
            tf.keras.callbacks.EarlyStopping(
                monitor=self.config.trainer.es_monitor,
                min_delta=self.config.trainer.es_delta,
                patience=self.config.trainer.es_epochs,
                verbose=1,
                mode='auto')
        )

        # Learning rate scheduler callback
        self.callbacks.append(tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler))

    def train(self, epochs=None):
        """Train the model using the tf.keras fit api call.
        """
        training_ds = self.data.training_ds(
            self.config.trainer.train_examples,
            self.config.trainer.batch_size
        )
        validation_ds = self.data.validation_ds(
            self.config.trainer.val_examples,
            self.config.trainer.batch_size
        )

        num_epochs = self.config.trainer.epochs
        if epochs is not None:
            num_epochs = epochs

        if self.config.trainer.steps_per_epoch == -1:
            steps = None
        else:
            steps = self.config.trainer.steps_per_epoch

        self.history = self.model.model.fit(
            training_ds,
            epochs=num_epochs,
            verbose=1,
            validation_data=validation_ds,
            callbacks=self.callbacks,
            steps_per_epoch=steps
        )

        return self.history

    def eval(self):
        """Run a quick evaluation.
        """
        testing_ds = self.data.testing_ds(
            self.config.trainer.test_examples,
            self.config.trainer.batch_size,
            strip=True
        )
        self.model.model.evaluate(testing_ds)

    def save(self):
        """Save the training history to file.
        """
        if self.history is not None:
            history_name = os.path.join(self.config.exp.exp_dir, 'history.csv')
            with open(history_name, mode='w') as history_file:
                writer = csv.DictWriter(history_file, self.history.history.keys())
                writer.writeheader()
                writer.writerow(self.history.history)
