"""Keras Functional API Model Implementations

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module contains the keras functional model definitions. All
are derived from the BaseModel class for ease-of-use in other parts
of the chips-cvn code
"""

import os
from tensorflow.keras import optimizers, Model, Input
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import tensorflow as tf
import sherpa


class BaseModel:
    """Base model class which all model implementations derive from."""
    def __init__(self, config):
        self.config = config
        self.build()

    def load(self, checkpoint_num=None):
        """Returns the correct model with its trained weights loaded."""
        if checkpoint_num is None:
            latest = tf.train.latest_checkpoint(self.config.exp_dir)
            self.model.load_weights(latest).expect_partial()
        else:
            checkpoint_path = self.config.exp_dir + "cp-" + str(checkpoint_num).zfill(4) + ".ckpt"
            self.model.load_weights(checkpoint_path).expect_partial()

    def summarise(self):
        """Prints model to stdout and plots diagram of model to file."""
        self.model.summary()  # Print the model structure to stdout

        # Plot an image of the model to file
        file_name = os.path.join(self.config.exp_dir, "model_diagram.png")
        tf.keras.utils.plot_model(self.model, to_file=file_name, show_shapes=True,
                                  show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

    def build(self):
        """Build the model, overide in derived model class."""
        raise NotImplementedError


class SingleParModel(BaseModel):
    """Single parameter estimation model class."""
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API."""
        inputs = Input(shape=self.config.img_shape, name='img')
        x = Conv2D(self.config.filters, self.config.kernel_size, padding='same', activation='relu')(inputs)
        x = Conv2D(self.config.filters, self.config.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.dropout)(x)
        x = Conv2D((self.config.filters*2), self.config.kernel_size, padding='same', activation='relu')(x)
        x = Conv2D((self.config.filters*2), self.config.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.dropout)(x)
        x = Flatten()(x)
        x = Dense(self.config.dense_units, activation='relu')(x)
        x = Dropout(self.config.dropout)(x)
        outputs = Dense(1, activation='linear', name=self.config.parameter)(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='single_par_model')

        self.loss = "mean_squared_error"
        self.loss_weights = None
        self.metrics = ["mae", "mse"]
        self.es_monitor = "val_mae"
        self.parameters = [self.config.parameter]

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.l_rate),
                           loss=self.loss,
                           loss_weights=self.loss_weights,
                           metrics=self.metrics)

    def study_parameters(self):
        """Returns the list of possible parameters for a SHERPA study."""
        pars = [
            sherpa.Ordinal(name='batch_size', range=self.config.batch_size),
            sherpa.Continuous(name='l_rate', range=self.config.l_rate, scale='log'),
            sherpa.Ordinal(name='dense_units', range=self.config.dense_units),
            sherpa.Continuous(name='dropout', range=self.config.dropout),
            sherpa.Ordinal(name='kernel_size', range=self.config.kernel_size),
            sherpa.Ordinal(name='filters', range=self.config.filters),
        ]
        return pars


class ClassificationModel(BaseModel):
    """Event classification model class."""
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API."""
        inputs = Input(shape=self.config.img_shape, name='img')
        x = Conv2D(self.config.filters, self.config.kernel_size, padding='same', activation='relu')(inputs)
        x = Conv2D(self.config.filters, self.config.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.dropout)(x)
        x = Conv2D((self.config.filters*2), self.config.kernel_size, padding='same', activation='relu')(x)
        x = Conv2D((self.config.filters*2), self.config.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.dropout)(x)
        x = Conv2D((self.config.filters*4), self.config.kernel_size, padding='same', activation='relu')(x)
        x = Conv2D((self.config.filters*4), self.config.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.dropout)(x)
        x = Flatten()(x)
        x = Dense(self.config.dense_units, activation='relu')(x)
        x = Dropout(self.config.dropout)(x)
        pdg_output = Dense(self.config.pdgs, activation='softmax', name="pdg")(x)
        type_output = Dense(self.config.types, activation='softmax', name="type")(x)
        self.model = Model(inputs=inputs, outputs=[pdg_output, type_output], name='classification_model')

        self.loss = {
            "pdg": "sparse_categorical_crossentropy",
            "type": "sparse_categorical_crossentropy",
        }
        self.loss_weights = {
            "pdg": 1.0,
            "type": 1.0
        }
        self.metrics = {
            "pdg": "accuracy",
            "type": "accuracy"
        }
        self.es_monitor = "val_accuracy"
        self.parameters = ["pdg", "type"]

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.l_rate),
                           loss=self.loss,
                           loss_weights=self.loss_weights,
                           metrics=self.metrics)

    def study_parameters(self):
        """Returns the list of possible parameters for a SHERPA study."""
        pars = [
            sherpa.Ordinal(name='batch_size', range=self.config.batch_size),
            sherpa.Continuous(name='l_rate', range=self.config.l_rate, scale='log'),
            sherpa.Ordinal(name='dense_units', range=self.config.dense_units),
            sherpa.Continuous(name='dropout', range=self.config.dropout),
            sherpa.Ordinal(name='kernel_size', range=self.config.kernel_size),
            sherpa.Ordinal(name='filters', range=self.config.filters),
        ]
        return pars


class MultiTaskModel(BaseModel):
    """Multi-task model class."""
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API."""
        inputs = Input(shape=self.config.img_shape, name='img')
        x = Conv2D(self.config.filters, self.config.kernel_size, padding='same', activation='relu')(inputs)
        x = Conv2D(self.config.filters, self.config.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.dropout)(x)
        x = Conv2D((self.config.filters*2), self.config.kernel_size, padding='same', activation='relu')(x)
        x = Conv2D((self.config.filters*2), self.config.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.dropout)(x)
        x = Conv2D((self.config.filters*4), self.config.kernel_size, padding='same', activation='relu')(x)
        x = Conv2D((self.config.filters*4), self.config.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.dropout)(x)
        x = Flatten()(x)
        x = Dense(self.config.dense_units, activation='relu')(x)
        x = Dropout(self.config.dropout)(x)
        pdg_path = Dense(self.config.dense_units, activation='relu')(x)
        type_path = Dense(self.config.dense_units, activation='relu')(x)
        energy_path = Dense(self.config.dense_units, activation='relu')(x)
        pdg_output = Dense(self.config.pdgs, activation='softmax', name="pdg")(pdg_path)
        type_output = Dense(self.config.types, activation='softmax', name="type")(type_path)
        energy_output = Dense(1, activation='linear', name="nuEnergy")(energy_path)
        self.model = Model(inputs=inputs, outputs=[pdg_output, type_output, energy_output], name='multi_task_model')

        self.loss = {
            "pdg": "sparse_categorical_crossentropy",
            "type": "sparse_categorical_crossentropy",
            "nuEnergy": "mean_squared_error",
        }
        self.loss_weights = {
            "pdg": 1.0,
            "type": 1.0,
            "nuEnergy": 1.0
        }
        self.metrics = {
            "pdg": "accuracy",
            "type": "accuracy",
            "nuEnergy": "mae"
        }
        self.es_monitor = "val_accuracy"
        self.parameters = ["pdg", "type", "nuEnergy"]

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.l_rate),
                           loss=self.loss,
                           loss_weights=self.loss_weights,
                           metrics=self.metrics)

    def study_parameters(self):
        """Returns the list of possible parameters for a SHERPA study."""
        pars = [
            sherpa.Ordinal(name='batch_size', range=self.config.batch_size),
            sherpa.Continuous(name='l_rate', range=self.config.l_rate, scale='log'),
            sherpa.Ordinal(name='dense_units', range=self.config.dense_units),
            sherpa.Continuous(name='dropout', range=self.config.dropout),
            sherpa.Ordinal(name='kernel_size', range=self.config.kernel_size),
            sherpa.Ordinal(name='filters', range=self.config.filters),
        ]
        return pars
