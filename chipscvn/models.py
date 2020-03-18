"""Keras Functional API Model Implementations

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module contains the keras functional model definitions. All
are derived from the BaseModel class for ease-of-use in other parts
of the chips-cvn code
"""

import os
from tensorflow.keras import optimizers, Model, Input
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, concatenate
import tensorflow as tf


class BaseModel:
    """Base model class which all model implementations derive from."""
    def __init__(self, config):
        self.config = config
        self.build()

    def load(self, checkpoint_num=None):
        """Returns the correct model with its trained weights loaded."""
        if checkpoint_num is None:
            latest = tf.train.latest_checkpoint(self.config.exp.checkpoints_dir)
            self.model.load_weights(latest).expect_partial()
        else:
            checkpoint_path = (
                self.config.exp.checkpoints_dir + 'cp-' + str(checkpoint_num).zfill(4) + '.ckpt')
            self.model.load_weights(checkpoint_path).expect_partial()

    def summarise(self):
        """Prints model to stdout and plots diagram of model to file."""
        self.model.summary()  # Print the model structure to stdout

        # Plot an image of the model to file
        file_name = os.path.join(self.config.exp.exp_dir, 'model_diagram.png')
        tf.keras.utils.plot_model(self.model, to_file=file_name, show_shapes=True,
                                  show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

    def build(self):
        """Build the model, overide in derived model class."""
        raise NotImplementedError


def GetModelBase(config):
    """Returns the base of the model that is shared"""
    inputs = []
    paths = []
    if config.model.reco_pars:
        vtxX_input = Input(shape=(1), name='r_vtxX')
        inputs.append(vtxX_input)
        paths.append(vtxX_input)

        vtxY_input = Input(shape=(1), name='r_vtxY')
        inputs.append(vtxY_input)
        paths.append(vtxY_input)

        vtxZ_input = Input(shape=(1), name='r_vtxZ')
        inputs.append(vtxZ_input)
        paths.append(vtxZ_input)

        dirTheta_input = Input(shape=(1), name='r_dirTheta')
        inputs.append(dirTheta_input)
        paths.append(dirTheta_input)

        dirPhi_input = Input(shape=(1), name='r_dirPhi')
        inputs.append(dirPhi_input)
        paths.append(dirPhi_input)

    images = config.data.img_size[2]
    shape = (config.data.img_size[0], config.data.img_size[1], 1)
    if config.data.stack:
        images = 1
        shape = config.data.img_size

    for channel in range(images):
        image_name = 'image_' + str(channel)
        image_input = Input(shape=shape, name=image_name)
        image_path = Conv2D(config.model.filters, config.model.kernel_size,
                            padding='same', activation='relu', name='conv1')(image_input)
        image_path = Conv2D(config.model.filters, config.model.kernel_size,
                            activation='relu', name='conv2')(image_path)
        image_path = MaxPooling2D(pool_size=2)(image_path)
        image_path = Dropout(config.model.dropout)(image_path)
        image_path = Conv2D((config.model.filters*2), config.model.kernel_size,
                            padding='same', activation='relu', name='conv3')(image_path)
        image_path = Conv2D((config.model.filters*2), config.model.kernel_size,
                            activation='relu', name='conv4')(image_path)
        image_path = MaxPooling2D(pool_size=2)(image_path)
        image_path = Dropout(config.model.dropout)(image_path)
        image_path = Conv2D((config.model.filters*4), config.model.kernel_size,
                            padding='same', activation='relu', name='conv5')(image_path)
        image_path = Conv2D((config.model.filters*4), config.model.kernel_size,
                            activation='relu', name='conv6')(image_path)
        image_path = MaxPooling2D(pool_size=2)(image_path)
        image_path = Dropout(config.model.dropout)(image_path)
        image_path = Conv2D((config.model.filters*8), config.model.kernel_size,
                            padding='same', activation='relu', name='conv7')(image_path)
        image_path = Conv2D((config.model.filters*8), config.model.kernel_size,
                            activation='relu', name='conv8')(image_path)
        image_path = MaxPooling2D(pool_size=2)(image_path)
        image_path = Dropout(config.model.dropout)(image_path)
        image_path = Flatten()(image_path)
        paths.append(image_path)
        inputs.append(image_input)

    x = concatenate(paths)
    x = Dense(config.model.dense_units, activation='relu')(x)
    x = Dense(config.model.dense_units, activation='relu')(x)
    x = Dense(config.model.dense_units, activation='relu', name='dense_final')(x)
    x = Dropout(config.model.dropout)(x)

    return x, inputs


class ParameterModel(BaseModel):
    """Single parameter estimation model class."""
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API."""

        x, inputs = GetModelBase(self.config)
        outputs = Dense(1, activation='linear', name=self.config.model.parameter)(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='parameter_model')

        self.loss = 'mean_squared_error'
        self.loss_weights = None
        self.metrics = ['mae', 'mse']
        self.es_monitor = 'val_mae'
        self.parameters = [self.config.model.parameter]

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           loss_weights=self.loss_weights,
                           metrics=self.metrics)


class CosmicModel(BaseModel):
    """Cosmic vs beam classification model class."""
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API."""

        x, inputs = GetModelBase(self.config)
        outputs = Dense(1, activation='sigmoid', name='t_cosmic_cat')(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='cosmic_model')

        self.loss = 'binary_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['true_cosmic']

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           metrics=self.metrics)


class BeamAllModel(BaseModel):
    """Beam all category classification model class."""
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API."""

        x, inputs = GetModelBase(self.config)
        outputs = Dense(18, activation='softmax', name='t_cat')(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='beam_all_model')
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['true_category']

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           metrics=self.metrics)


class BeamFullCombModel(BaseModel):
    """Beam full combined category classification model class."""
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API."""

        x, inputs = GetModelBase(self.config)
        outputs = Dense(4, activation='softmax', name='t_full_cat')(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='beam_full_comb_model')
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['true_category']

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           metrics=self.metrics)


class BeamNuNCCombModel(BaseModel):
    """Beam Nu NC category combined classification model class."""
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API."""

        x, inputs = GetModelBase(self.config)
        outputs = Dense(13, activation='softmax', name='t_nu_nc_cat')(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='beam_nu_nc_comb_model')
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['true_category']

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           metrics=self.metrics)


class BeamNCCombModel(BaseModel):
    """Beam NC category combined classification model class."""
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API."""

        x, inputs = GetModelBase(self.config)
        outputs = Dense(10, activation='softmax', name='t_nc_cat')(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='beam_nc_comb_model')
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['true_category']

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           metrics=self.metrics)


class BeamMultiModel(BaseModel):
    """Beam Multi-task model class."""
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API."""

        x, inputs = GetModelBase(self.config)
        category_path = Dense(self.config.model.dense_units, activation='relu')(x)
        energy_path = Dense(self.config.model.dense_units, activation='relu')(x)
        category_output = Dense(18, activation='softmax', name='t_cat')(category_path)
        energy_output = Dense(1, activation='linear', name='t_nuEnergy')(energy_path)
        self.model = Model(inputs=inputs,
                           outputs=[category_output, energy_output],
                           name='beam_multi_model')

        self.loss = {
            't_cat': 'sparse_categorical_crossentropy',
            't_nuEnergy': 'mean_squared_error',
        }
        self.loss_weights = {
            't_cat': 1.0,
            't_nuEnergy': 1.0
        }
        self.metrics = {
            't_cat': 'accuracy',
            't_nuEnergy': 'mae'
        }
        self.es_monitor = 'val_accuracy'
        self.parameters = ['t_cat', 't_nuEnergy']

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           loss_weights=self.loss_weights,
                           metrics=self.metrics)
