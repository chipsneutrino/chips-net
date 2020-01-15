"""Keras Functional API Model Implementations

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module contains the keras functional model definitions. All
are derived from the BaseModel class for ease-of-use in other parts
of the chips-cvn code
"""

import os
from tensorflow import keras
from tensorflow.keras import optimizers, callbacks, Model, Input
from tensorflow.keras.layers import (Dense, Dropout, Conv2D,
                                     MaxPooling2D, Flatten)


class BaseModel:
    """Base model class which all implementations derive from."""
    def __init__(self, conf):
        self.conf = conf

    def init(self, pars):
        """Initialises the model."""
        '''
        We take in a parameter namespace to use in initialising the model
        7) Early stopping delta (es_delta) *
        8) Early stopping epochs (es_epochs) *
        '''

    def compile(self):
        """Compiles the model."""
        self.model.compile(optimizer=optimizers.Adam(
                           learning_rate=self.conf.l_rate),
                           loss=self.loss, metrics=self.metrics)

        self.model.summary()  # Print the model summary

        # Plot an image of the model to file
        file_name = os.path.join(self.conf.exp_dir, "plot.png")
        keras.utils.plot_model(self.model, to_file=file_name, show_shapes=True,
                               show_layer_names=True, rankdir='TB',
                               expand_nested=False, dpi=96)

    def fit(self, train_ds, val_ds):
        """Given a training and validation dataset, fit the model."""
        callbacks_list = []

        checkpoint = callbacks.ModelCheckpoint(filepath=self.conf.exp_dir,
                                               verbose=0)
        callbacks_list.append(checkpoint)

        tensorboard = callbacks.TensorBoard(log_dir="tmp", histogram_freq=1)
        callbacks_list.append(tensorboard)

        stopping = callbacks.EarlyStopping(monitor=self.es_monitor,
                                           min_delta=self.conf.es_delta,
                                           patience=self.conf.es_epochs,
                                           verbose=1, mode='min')
        callbacks_list.append(stopping)

        train_ds = train_ds.batch(self.conf.batch_size, drop_remainder=True)
        val_ds = val_ds.batch(self.conf.batch_size, drop_remainder=True)

        self.history = self.model.fit(train_ds,
                                      epochs=self.conf.train_epochs,
                                      verbose=1,
                                      validation_data=val_ds,
                                      callbacks=callbacks_list)

    def evaluate(self, test_ds):
        """Evaluate the trained model on a test dataset."""
        test_ds = test_ds.batch(self.conf.batch_size, drop_remainder=True)
        self.model.evaluate(test_ds)


class PIDModel(BaseModel):
    """PID event categorisation model class."""
    def __init__(self, conf):
        super().__init__(conf)

    def build(self):
        """Builds the model using the keras functional API."""
        inputs = Input(shape=self.conf.img_shape, name='img')
        x = Conv2D(filters=64, kernel_size=3, padding='same',
                   activation='relu')(inputs)
        x = Conv2D(filters=64, kernel_size=3, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(filters=128, kernel_size=3, padding='same',
                   activation='relu')(x)
        x = Conv2D(filters=128, kernel_size=3, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(filters=256, kernel_size=3, padding='same',
                   activation='relu')(x)
        x = Conv2D(filters=256, kernel_size=3, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.conf.categories, activation='softmax')(x)
        self.model = Model(inputs=inputs,
                           outputs=outputs,
                           name='pid_model')

        self.loss = "sparse_categorical_crossentropy"
        self.metrics = ["accuracy"]
        self.es_monitor = "val_acc"


class SingleParModel(BaseModel):
    """Parameter estimation model class."""
    def __init__(self, conf):
        super().__init__(conf)

    def build(self):
        """Builds the model using the keras functional API."""
        inputs = Input(shape=self.conf.img_shape, name='img')
        x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
        x = Conv2D(filters=32, kernel_size=3, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=3, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.0)(x)
        outputs = Dense(1, activation='linear', name=self.conf.parameter)(x)
        self.model = Model(inputs=inputs,
                           outputs=outputs,
                           name='ppe_model')

        self.loss = "mean_squared_error"
        self.metrics = ["mae", "mse"]
        self.es_monitor = "val_mae"
