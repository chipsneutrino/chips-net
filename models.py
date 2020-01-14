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
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization,
                                     Conv2D, MaxPooling2D, AveragePooling2D,
                                     concatenate, Flatten)
from tensorflow.keras.regularizers import l2


class BaseModel:
    """Base model class which all implementations derive from."""
    def __init__(self, config):
        self.config = config

    def summary(self):
        """Prints the model summary."""
        self.model.summary()

    def plot(self):
        """Plots an image of the model to file."""
        file_name = os.path.join("experiments", self.config.exp_name,
                                 "summary/plot.png")
        keras.utils.plot_model(self.model, to_file=file_name, show_shapes=True,
                               show_layer_names=True, rankdir='TB',
                               expand_nested=False, dpi=96)

    def compile(self):
        """Compiles the model."""
        self.model.compile(optimizer=optimizers.Adam(
                           learning_rate=self.config.learning_rate),
                           loss=self.loss, metrics=self.metrics)

    def fit(self, train_ds, val_ds):
        """Given a training and validation dataset, fit the model."""
        callbacks_list = []

        dir = os.path.join("experiments", self.config.exp_name, "checkpoint")
        checkpoint = callbacks.ModelCheckpoint(filepath=dir, verbose=0)
        callbacks_list.append(checkpoint)

        tensorboard = callbacks.TensorBoard(log_dir="tmp", histogram_freq=1)
        callbacks_list.append(tensorboard)

        stopping = callbacks.EarlyStopping(monitor=self.es_monitor,
                                           min_delta=self.config.es_delta,
                                           patience=self.config.es_epochs,
                                           verbose=1, mode='min')
        callbacks_list.append(stopping)

        self.history = self.model.fit(train_ds,
                                      epochs=self.config.epochs,
                                      verbose=1,
                                      validation_data=val_ds,
                                      callbacks=callbacks_list)

    def evaluate(self, test_ds):
        """Evaluate the trained model on a test dataset."""
        self.model.evaluate(test_ds)


class PIDModel(BaseModel):
    """PID event categorisation model class."""
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API."""
        inputs = Input(shape=(self.config.img_size, self.config.img_size, 3),
                       name='img')
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
        outputs = Dense(self.config.categories, activation='softmax')(x)
        self.model = Model(inputs=inputs,
                           outputs=outputs,
                           name='pid_model')

        self.loss = "sparse_categorical_crossentropy"
        self.metrics = ["accuracy"]
        self.es_monitor = "val_acc"


class PPEModel(BaseModel):
    """Parameter estimation model class."""
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API."""
        inputs = Input(shape=(self.config.img_size, self.config.img_size, 3), name='img')
        x = Conv2D(filters=32, kernel_size=7, strides=2, padding="same", activation='relu', name="conv1")(x)
        x = BatchNormalization(axis=3, name="bn1")(x)
    
        x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
        x = Conv2D(filters=32, kernel_size=3, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=3, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.0)(x)
        outputs = Dense(1, activation='linear')(x)
        self.model = Model(inputs=inputs,
                           outputs=outputs,
                           name='ppe_model')

        self.loss = "mean_squared_error"
        self.metrics = ["mae", "mse"]
        self.es_monitor = "val_mae"


class InceptionModel(BaseModel):
    """Parameter estimation inception based model class."""
    def __init__(self, config):
        super().__init__(config)

    def Conv2d_All(self, x, nb_filter, kernel_size, padding='same',
                   strides=(1, 1), name=None):
        """Defines the a convolutional layer with a batch normalisation."""
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides,
                   activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)

        return x

    def Inception(self, x, nb_filter):
        """Defines the inception module."""
        b1x1 = self.Conv2d_All(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
        b3x3 = self.Conv2d_All(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
        b3x3 = self.Conv2d_All(b3x3, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)
        b5x5 = self.Conv2d_All(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
        b5x5 = self.Conv2d_All(b5x5, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
        bpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        bpool = self.Conv2d_All(bpool, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
        x = concatenate([b1x1, b3x3, b5x5, bpool], axis=3)
        return x

    def build(self):
        """Builds the model using the keras functional API."""
        input = Input(shape=(self.config.img_size, self.config.img_size, 3), dtype='float32', name='input')
        x = self.Conv2d_All(input, 64, (7, 7), strides=(2, 2), padding='same')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = self.Conv2d_All(x, 192, (3, 3), strides=(1, 1), padding='same')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = self.Inception(x, 64)
        x = Dropout(0.2)(x)
        x = self.Inception(x, 120)
        x = Dropout(0.2)(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = self.Inception(x, 128)
        x = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.1))(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)
        out = Dense(1, activation='linear', name='out')(x)
        self.model = Model(inputs=input, outputs=[out], name='inception_model')

        self.loss = "mean_squared_error"
        self.metrics = ["mae", "mse"]
        self.es_monitor = "val_mae"
