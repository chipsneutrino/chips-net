# -*- coding: utf-8 -*-

"""Training classes for fitting all the models

This module contains all the training classes used to fit the different
models. All are derived from the BaseTrainer class and allow for
customisation of the training loop if required.
"""

import os
import csv
import tensorflow as tf
from tqdm import tqdm


def get_trainer(config, model, data):
    """Returns the correct trainer for the configuration.
    Args:
        config (dotmap.DotMap): DotMap Configuration namespace
        model (chipsnet.models model): Model to use with trainer
        data (chipsnet.data.Loader): Data loader
    Returns:
        chipsnet.trainers trainer: Training class
    """
    if config.model.name == "multi":
        return MultiTrainer(config, model, data)
    else:
        return BasicTrainer(config, model, data)


class BaseTrainer(object):
    """Base model class which all implementations derive from.
    """
    def __init__(self, config, model, data):
        """Initialise the BaseTrainer.
        Args:
            config (dotmap.DotMap): DotMap Configuration namespace
            model (chipsnet.models model): Model to use with trainer
            data (chipsnet.data.Loader): Data loader
        """
        self.config = config
        self.model = model
        self.data = data
        self.callbacks = []
        self.history = None

    def train(self):
        """Train the model, overide in derived model class.
        """
        raise NotImplementedError

    def eval(self):
        """Run a quick evaluation, overide in derived model class.
        """
        raise NotImplementedError

    def save(self):
        """Save the training history to file, overide in derived model class.
        """
        raise NotImplementedError


class BasicTrainer(BaseTrainer):
    """Basic trainer class, implements a simple keras fitter.
    """
    def __init__(self, config, model, data):
        """Initialise the BasicTrainer.
        Args:
            config (dotmap.DotMap): DotMap Configuration namespace
            model (chipsnet.models model): Model to use with trainer
            data (chipsnet.data.Loader): Data loader
        """
        super().__init__(config, model, data)
        self.init_callbacks()

    def lr_scheduler(self, epoch):
        """Learning rate schedule function.
        Args:
            epoch (int): Epoch number
        Returns:
            float: The generated dataset
        """
        lr = self.config.model.lr * 1/(1 + self.config.model.lr_decay * epoch)
        print("\nLearning Rate: {}".format(lr))
        return lr

    def init_callbacks(self):
        """Initialise the keras callbacks to be called at the end of each epoch.
        """
        self.callbacks.append(  # Tensorboard callback for viewing plots of model training
            tf.keras.callbacks.TensorBoard(
                log_dir=self.config.exp.tensorboard_dir,
                histogram_freq=1,
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
                monitor=self.model.es_monitor,
                min_delta=self.config.trainer.es_delta,
                patience=self.config.trainer.es_epochs,
                verbose=1,
                mode='auto')
        )

        # Learning rate scheduler callback
        self.callbacks.append(tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler))

    def train(self, additional_callbacks=[]):
        """Train the model using the tf.keras fit api call.
        """
        self.callbacks.extend(additional_callbacks)
        training_dataset = self.data.training_ds
        validation_dataset = self.data.validation_ds

        if self.config.trainer.steps_per_epoch == -1:
            steps = None
        else:
            steps = self.config.trainer.steps_per_epoch

        self.history = self.model.model.fit(
            training_dataset,
            epochs=self.config.trainer.epochs,
            verbose=1,
            validation_data=validation_dataset,
            callbacks=self.callbacks,
            steps_per_epoch=steps
        )

    def eval(self):
        """Run a quick evaluation.
        """
        self.model.model.evaluate(self.data.testing_ds)

    def save(self):
        """Save the training history to file.
        """
        if self.history is not None:
            history_name = os.path.join(self.config.exp.exp_dir, 'history.csv')
            with open(history_name, mode='w') as history_file:
                writer = csv.DictWriter(history_file, self.history.history.keys())
                writer.writeheader()
                writer.writerow(self.history.history)


class MultiTrainer(BaseTrainer):
    """CHIPS multi-task trainer class, implements a custom training loop.
    """
    def __init__(self, config, model, data):
        """Initialise the MultiTrainer.
        Args:
            config (dotmap.DotMap): DotMap Configuration namespace
            model (chipsnet.models model): Model to use with trainer
            data (chipsnet.data.Loader): Data loader
        """
        super().__init__(config, model, data)

    def train(self):
        """Train the model using a custom loop.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.model.lr)
        training_ds = self.data.training_ds

        loss_results = []
        for epoch in range(self.config.trainer.epochs):
            epoch_l_metric = tf.keras.metrics.Mean()
            epoch_c_metric = tf.keras.metrics.SparseCategoricalAccuracy()
            epoch_e_metric = tf.keras.metrics.MeanAbsoluteError()

            # Iterate over the batches of the dataset.
            for step, (inputs, labels) in enumerate(tqdm(
                    training_ds,
                    total=self.config.trainer.steps_per_epoch,
                    desc='Epoch {}:'.format(epoch))):

                # Run forward pass a record on the GradientTape
                with tf.GradientTape() as tape:
                    output = self.model.model(inputs, training=True)
                    loss = sum(self.model.model.losses)

                # Use the gradient tape to retrieve gradients wrt the loss
                grads = tape.gradient(loss, self.model.model.trainable_weights)

                # Run gradient descent by updating variables to minimise the loss
                optimizer.apply_gradients(zip(grads, self.model.model.trainable_weights))

                # Update metrics
                epoch_l_metric(loss)
                epoch_c_metric(labels['t_all_cat'], output[0])
                epoch_e_metric(labels['t_nuEnergy'], output[1])

            loss_results.append(epoch_l_metric.result())
            print('Loss: {}, Category accuracy: {}, Energy MAE: {}'.format(
                epoch_l_metric.result(), epoch_c_metric.result(), epoch_e_metric.result()))

    def eval(self):
        """Run a quick evaluation.
        """
        print('eval not yet implemented')

    def save(self):
        """Save the training history to file.
        """
        print('save not yet implemented')
