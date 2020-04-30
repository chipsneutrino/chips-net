# -*- coding: utf-8 -*-

"""Training classes for fitting all the models

This module contains all the training classes used to fit the different
models. All are derived from the BaseTrainer class and allow for
customisation of the training loop if required.
"""

import os
import csv
import tensorflow as tf


def get_trainer(config, model, data, strategy):
    """Returns the correct trainer for the configuration.
    Args:
        config (dotmap.DotMap): DotMap Configuration namespace
        model (chipscvn.models model): Model to use with trainer
        data (chipscvn.data.DataLoader): Data loader
    Returns:
        chipscvn.trainers trainer: Training class
    """
    if config.model.name == "multi":
        # return MultiTrainer(config, model, data, strategy)
        return None
    else:
        return BasicTrainer(config, model, data, strategy)


class BaseTrainer(object):
    """Base model class which all implementations derive from.
    """
    def __init__(self, config, model, data, strategy):
        """Initialise the BaseTrainer.
        Args:
            config (dotmap.DotMap): DotMap Configuration namespace
            model (chipscvn.models model): Model to use with trainer
            data (chipscvn.data.DataLoader): Data loader
        """
        self.config = config
        self.model = model
        self.data = data
        self.strategy = strategy
        self.callbacks = []
        self.history = []

    def train(self):
        """Train the model, overide in derived model class.
        """
        raise NotImplementedError


class BasicTrainer(BaseTrainer):
    """Basic trainer class, implements a simple keras fitter.
    """
    def __init__(self, config, model, data, strategy):
        """Initialise the BasicTrainer.
        Args:
            config (dotmap.DotMap): DotMap Configuration namespace
            model (chipscvn.models model): Model to use with trainer
            data (chipscvn.data.DataLoader): Data loader
        """
        super().__init__(config, model, data, strategy)
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

        self.model.model.evaluate(self.data.testing_ds)
        self.save()

    def save(self):
        """Saves the training history to file.
        """
        history_name = os.path.join(self.config.exp.exp_dir, 'history.csv')
        with open(history_name, mode='w') as history_file:
            writer = csv.DictWriter(history_file, self.history.history.keys())
            writer.writeheader()
            writer.writerow(self.history.history)

'''
class MultiTrainer(BaseTrainer):
    """CHIPS multi-task trainer class, implements a custom training loop.
    """
    def __init__(self, config, model, data):
        """Initialise the MultiTrainer.
        Args:
            config (dotmap.DotMap): DotMap Configuration namespace
            model (chipscvn.models model): Model to use with trainer
            data (chipscvn.data.DataLoader): Data loader
        """
        super().__init__(config, model, data)
        self.epochs = config.trainer.epochs
        self.batch_size = config.data.batch_size
        self.enable_function = True
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.model.lr)
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.train_c_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_c_acc')
        self.train_e_metric = tf.keras.metrics.MeanAbsoluteError(name='train_e_mae')
        self.val_loss_metric = tf.keras.metrics.Sum(name='val_loss')
        self.val_c_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_c_acc')
        self.val_e_metric = tf.keras.metrics.MeanAbsoluteError(name='val_e_mae')
        self.dist_train_ds = self.strategy.experimental_distribute_dataset(self.data.training_ds)
        self.val_train_ds = self.strategy.experimental_distribute_dataset(self.data.validation_ds)

    def train_step(self, batch):
        """One train step.
        Args:
            batch: one batch input.
        Returns:
            loss: Scaled loss.
        """
        inputs, labels = batch
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = sum(self.model.model.losses)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_acc_metric(label, predictions)
        return loss

    def test_step(self, batch):
        """One test step.
        Args:
            batch: one batch input.
        """
        inputs, labels = batch
        predictions = self.model(inputs, training=False)

        unscaled_test_loss = self.loss_object(label, predictions) + sum(
            self.model.losses)

        self.test_acc_metric(label, predictions)
        self.test_loss_metric(unscaled_test_loss)

    def train(self):
        """Train the model using a custom loop.
        """
        loss_results = []
        for epoch in range(self.config.trainer.epochs):
            print('Start of epoch {}'.format(epoch))
            epoch_loss_avg = tf.keras.metrics.Mean()

            # Iterate over the batches of the dataset.
            for step, dist_inputs in enumerate(dist_training_ds):
                train_step(dist_inputs)

                model_input = [inputs['image_0'], labels[self.config.model.category], labels['t_nuEnergy']]

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables autodifferentiation.
                with tf.GradientTape() as tape:
                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    c_pred, e_pred = self.model.model(model_input, training=True)  # Logits for this minibatch

                    loss = sum(self.model.model.losses)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss, self.model.model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, self.model.model.trainable_weights))

                # Track progress
                epoch_loss_avg.update_state(loss)  # Add current batch loss

            loss_results.append(epoch_loss_avg.result())
            print('Epoch: {}, Loss: {}'.format(epoch, epoch_loss_avg.result()))

        @tf.function
        def train_step(dist_inputs):
            def step_fn(inputs):
                features, labels = inputs

                with tf.GradientTape() as tape:
                    logits = self.model.model(features, training=True)

                    output = self.model.model(model_input, training=True)  # Logits for this minibatch
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits, labels=labels)
                    loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)

                grads = tape.gradient(loss, self.model.model.trainable_variables)
                optimizer.apply_gradients(list(zip(grads, self.model.model.trainable_variables)))
                return cross_entropy

            per_example_losses = self.strategy.run(step_fn, args=(dist_inputs,))
            mean_loss = self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
            return mean_loss
'''
