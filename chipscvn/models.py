# -*- coding: utf-8 -*-

"""
tf.keras Functional API Model Implementations

This module contains the keras functional model definitions. All
are derived from the BaseModel class for ease-of-use in other parts
of the chips-cvn code
"""

import os
from tensorflow.keras import optimizers, Model, Input
from tensorflow.keras.layers import (Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D,
                                     Flatten, concatenate, BatchNormalization)
from tensorflow.keras.regularizers import l2
import tensorflow as tf


class BaseModel:

    """
    Base model class which all model implementations derive from.
    """

    def __init__(self, config):
        """
        Initialise the BaseModel.

        Args:
            config (str): Dotmap configuration namespace
        """
        self.config = config
        self.build()

    def load(self, checkpoint_num=None):
        """
        Returns the correct model with its trained weights loaded.

        Args:
            checkpoint_num (int): Checkpoint number to use
        """
        if checkpoint_num is None:
            latest = tf.train.latest_checkpoint(self.config.exp.checkpoints_dir)
            self.model.load_weights(latest).expect_partial()
        else:
            checkpoint_path = (
                self.config.exp.checkpoints_dir + 'cp-' + str(checkpoint_num).zfill(4) + '.ckpt')
            self.model.load_weights(checkpoint_path).expect_partial()

    def summarise(self):
        """
        Prints model summary to stdout and plots diagram of model to file.
        """
        self.model.summary()  # Print the model structure to stdout

        # Plot an image of the model to file
        file_name = os.path.join(self.config.exp.exp_dir, 'model_diagram.png')
        tf.keras.utils.plot_model(
            self.model,
            to_file=file_name,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=96
        )

    def build(self):
        """
        Build the model, overide in derived model class.
        """
        raise NotImplementedError


def GetModelBase(config):
    """
    Returns the base model that is shared.

    Args:
        config (dotmap.DotMap): Configuration namespace
    Returns:
        tf.tensor: Keras functional tensor
    """
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
        conv_name = 'conv_' + str(channel) + "_"
        image_input = Input(shape=shape, name=image_name)
        image_path = Conv2D(config.model.filters, config.model.kernel_size,
                            padding='same', activation='relu', name=(conv_name + "1"))(image_input)
        image_path = Conv2D(config.model.filters, config.model.kernel_size,
                            activation='relu', name=(conv_name + "2"))(image_path)
        image_path = MaxPooling2D(pool_size=2)(image_path)
        image_path = Dropout(config.model.dropout)(image_path)
        image_path = Conv2D((config.model.filters*2), config.model.kernel_size,
                            padding='same', activation='relu', name=(conv_name + "3"))(image_path)
        image_path = Conv2D((config.model.filters*2), config.model.kernel_size,
                            activation='relu', name=(conv_name + "4"))(image_path)
        image_path = MaxPooling2D(pool_size=2)(image_path)
        image_path = Dropout(config.model.dropout)(image_path)
        image_path = Conv2D((config.model.filters*4), config.model.kernel_size,
                            padding='same', activation='relu', name=(conv_name + "5"))(image_path)
        image_path = Conv2D((config.model.filters*4), config.model.kernel_size,
                            activation='relu', name=(conv_name + "6"))(image_path)
        image_path = MaxPooling2D(pool_size=2)(image_path)
        image_path = Dropout(config.model.dropout)(image_path)
        image_path = Conv2D((config.model.filters*8), config.model.kernel_size,
                            padding='same', activation='relu', name=(conv_name + "7"))(image_path)
        image_path = Conv2D((config.model.filters*8), config.model.kernel_size,
                            activation='relu', name=(conv_name + "8"))(image_path)
        image_path = MaxPooling2D(pool_size=2)(image_path)
        image_path = Dropout(config.model.dropout)(image_path)
        image_path = Flatten()(image_path)
        paths.append(image_path)
        inputs.append(image_input)

    if len(paths) == 1:
        x = paths[0]
    else:
        x = concatenate(paths)

    x = Dense(config.model.dense_units, activation='relu')(x)
    x = Dense(config.model.dense_units, activation='relu')(x)
    x = Dense(config.model.dense_units, activation='relu', name='dense_final')(x)
    x = Dropout(config.model.dropout)(x)
    return x, inputs


def Conv2d_All(x, nb_filter, kernel_size, padding='same', strides=(1, 1)):
    """
    Returns Conv2d with Batch Normalisation.

    Args:
        x (tf.tensor): Input tensor
        nb_filter (int): Number of convolutional filters
        kernel_size (int): Size of convolutional kernal
        padding (str): Convolutional padding
        strides (tuple(int, int)): Stride size when applying convolution
    Returns:
        tf.tensor: Keras functional tensor
    """
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu')(x)
    x = BatchNormalization(axis=3)(x)
    return x


def Inception(x, nb_filter):
    """
    Returns a Keras functional API Inception Module.

    Args:
        x (tf.tensor): Input tensor
        nb_filter (int): Number of convolutional filters
    Returns:
        tf.tensor: Keras functional inceptional module tensor
    """
    b1x1 = Conv2d_All(x, nb_filter, (1, 1), padding='same', strides=(1, 1))
    b3x3 = Conv2d_All(x, nb_filter, (1, 1), padding='same', strides=(1, 1))
    b3x3 = Conv2d_All(b3x3, nb_filter, (3, 3), padding='same', strides=(1, 1))
    b5x5 = Conv2d_All(x, nb_filter, (1, 1), padding='same', strides=(1, 1))
    b5x5 = Conv2d_All(b5x5, nb_filter, (1, 1), padding='same', strides=(1, 1))
    bpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    bpool = Conv2d_All(bpool, nb_filter, (1, 1), padding='same', strides=(1, 1))
    x = concatenate([b1x1, b3x3, b5x5, bpool], axis=3)
    return x


def InceptionBase(config):
    """
    Returns the Inception base of the model that is shared.

    Args:
        config (dotmap.DotMap): Configuration namespace
    Returns:
        tuple(tf.tensor, List[tf.tensor]): (Keras functional tensor, List of input layers)
    """
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
        image_path = Conv2d_All(image_input, 64, (7, 7), strides=(2, 2), padding='same')
        image_path = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(image_path)
        image_path = Conv2d_All(image_path, 192, (3, 3), strides=(1, 1), padding='same')
        image_path = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(image_path)
        image_path = Inception(image_path, 64)
        image_path = Inception(image_path, 120)
        image_path = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(image_path)
        image_path = Inception(image_path, 128)
        image_path = Inception(image_path, 128)
        image_path = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(image_path)
        image_path = Inception(image_path, 256)
        image_path = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(image_path)
        image_path = Flatten()(image_path)
        image_path = Dense(1024, activation='relu', kernel_regularizer=l2(0.1))(image_path)
        paths.append(image_path)
        inputs.append(image_input)

    if len(paths) == 1:
        x = paths[0]
    else:
        x = concatenate(paths)

    x = Dense(1024, activation='relu')(x)
    x = Dense(config.model.dense_units, activation='relu')(x)
    x = Dense(config.model.dense_units, activation='relu', name='dense_final')(x)
    return x, inputs


class ParameterModel(BaseModel):

    """
    Single parameter estimation model class.
    """

    def __init__(self, config):
        """
        Initialise the ParameterModel.

        Args:
            config (str): Dotmap configuration namespace
        """
        super().__init__(config)

    def build(self):
        """
        Builds the model using the keras functional API.
        """
        x, inputs = GetModelBase(self.config)
        outputs = Dense(1, activation='linear', name=self.config.model.parameter)(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='parameter_model')

        self.loss = 'mean_squared_error'
        self.metrics = ['mae', 'mse']
        self.es_monitor = 'val_mae'
        self.parameters = [self.config.model.parameter]

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           metrics=self.metrics)


class CosmicModel(BaseModel):

    """
    Cosmic vs beam classification model class.
    """

    def __init__(self, config):
        """
        Initialise the CosmicModel.

        Args:
            config (str): Dotmap configuration namespace
        """
        super().__init__(config)

    def build(self):
        """
        Builds the model using the keras functional API.
        """
        x, inputs = GetModelBase(self.config)
        outputs = Dense(1, activation='sigmoid', name='t_cosmic_cat')(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='cosmic_model')

        self.loss = 'binary_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['t_cosmic_cat']

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           metrics=self.metrics)


class BeamAllModel(BaseModel):

    """
    Beam all category classification model class.
    """

    def __init__(self, config):
        """
        Initialise the BeamAllModel.

        Args:
            config (str): Dotmap configuration namespace
        """
        super().__init__(config)

    def build(self):
        """
        Builds the model using the keras functional API.
        """
        self.categories = 16
        self.cat = 't_cat'
        x, inputs = GetModelBase(self.config)
        outputs = Dense(self.categories, activation='softmax', name=self.cat)(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='beam_all_model')
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['t_cat']

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           metrics=self.metrics)

        self.labels = ["EL-CC-QEL", "EL-CC-RES", "EL-CC-DIS", "EL-CC-COH",
                       "MU-CC-QEL", "MU-CC-RES", "MU-CC-DIS", "MU-CC-COH",
                       "EL-NC-QEL", "EL-NC-RES", "EL-NC-DIS", "EL-NC-COH",
                       "MU-NC-QEL", "MU-NC-RES", "MU-NC-DIS", "MU-NC-COH"]

    def combine_outputs(self, ev):
        """
        Combine outputs into fully combined categories.

        Args:
            ev (dict): Pandas single event(row) dict
        Returns:
            List[float, float, float]: List of combined category outputs
        """
        nuel = (ev['b_out_0'] + ev['b_out_1'] + ev['b_out_2'] + ev['b_out_3'])
        numu = (ev['b_out_4'] + ev['b_out_5'] + ev['b_out_6'] + ev['b_out_7'])
        nc = (ev['b_out_8'] + ev['b_out_9'] + ev['b_out_10'] + ev['b_out_11'] +
              ev['b_out_12'] + ev['b_out_13'] + ev['b_out_14'] + ev['b_out_15'])
        return [nuel, numu, nc]


class BeamFullCombModel(BaseModel):

    """
    Beam full combined category classification model class.
    """

    def __init__(self, config):
        """
        Initialise the BeamFullCombModel.

        Args:
            config (str): Dotmap configuration namespace
        """
        super().__init__(config)

    def build(self):
        """
        Builds the model using the keras functional API.
        """
        self.categories = 3
        self.cat = 't_full_cat'
        x, inputs = GetModelBase(self.config)
        outputs = Dense(self.categories, activation='softmax', name=self.cat)(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='beam_full_comb_model')
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['t_full_cat']

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           metrics=self.metrics)

        self.labels = ["EL-CC", "MU-CC", "NC"]

    def combine_outputs(self, ev):
        """
        Combine outputs into fully combined categories.

        Args:
            ev (dict): Pandas single event(row) dict
        Returns:
            List[float, float, float]: List of combined category outputs
        """
        nuel = ev['b_out_0']
        numu = ev['b_out_1']
        nc = ev['b_out_2']
        return [nuel, numu, nc]


class BeamNuNCCombModel(BaseModel):

    """
    Beam Nu NC category combined classification model class.
    """

    def __init__(self, config):
        """
        Initialise the BeamNuNCCombModel.

        Args:
            config (str): Dotmap configuration namespace
        """
        super().__init__(config)

    def build(self):
        """
        Builds the model using the keras functional API.
        """
        self.categories = 12
        self.cat = 't_nu_nc_cat'
        x, inputs = GetModelBase(self.config)
        outputs = Dense(self.categories, activation='softmax', name=self.cat)(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='beam_nu_nc_comb_model')
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['t_nu_nc_cat']

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           metrics=self.metrics)

        self.labels = ["EL-CC-QEL", "EL-CC-RES", "EL-CC-DIS", "EL-CC-COH",
                       "MU-CC-QEL", "MU-CC-RES", "MU-CC-DIS", "MU-CC-COH",
                       "NC-QEL", "NC-RES", "NC-DIS", "NC-COH"]

    def combine_outputs(self, ev):
        """
        Combine outputs into fully combined categories.

        Args:
            ev (dict): Pandas single event(row) dict
        Returns:
            List[float, float, float]: List of combined category outputs
        """
        nuel = (ev['b_out_0'] + ev['b_out_1'] + ev['b_out_2'] + ev['b_out_3'])
        numu = (ev['b_out_4'] + ev['b_out_5'] + ev['b_out_6'] + ev['b_out_7'])
        nc = (ev['b_out_8'] + ev['b_out_9'] + ev['b_out_10'] + ev['b_out_11'])
        return [nuel, numu, nc]


class BeamNCCombModel(BaseModel):

    """
    Beam NC category combined classification model class.
    """

    def __init__(self, config):
        """
        Initialise the BeamNCCombModel.

        Args:
            config (str): Dotmap configuration namespace
        """
        super().__init__(config)

    def build(self):
        """
        Builds the model using the keras functional API.
        """
        self.categories = 9
        self.cat = 't_nc_cat'
        x, inputs = GetModelBase(self.config)
        outputs = Dense(self.categories, activation='softmax', name=self.cat)(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='beam_nc_comb_model')
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['t_nc_cat']

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           metrics=self.metrics)

        self.labels = ["EL-CC-QEL", "EL-CC-RES", "EL-CC-DIS", "EL-CC-COH",
                       "MU-CC-QEL", "MU-CC-RES", "MU-CC-DIS", "MU-CC-COH",
                       "NC"]

    def combine_outputs(self, ev):
        """
        Combine outputs into fully combined categories.

        Args:
            ev (dict): Pandas single event(row) dict
        Returns:
            List[float, float, float]: List of combined category outputs
        """
        nuel = (ev['b_out_0'] + ev['b_out_1'] + ev['b_out_2'] + ev['b_out_3'])
        numu = (ev['b_out_4'] + ev['b_out_5'] + ev['b_out_6'] + ev['b_out_7'])
        nc = ev['b_out_8']
        return [nuel, numu, nc]


class BeamMultiModel(BaseModel):

    """
    Beam Multi-task model class.
    """

    def __init__(self, config):
        """
        Initialise the BeamMultiModel.

        Args:
            config (str): Dotmap configuration namespace
        """
        super().__init__(config)

    def build(self):
        """
        Builds the model using the keras functional API.
        """
        self.categories = 16
        self.cat = 't_cat'
        x, inputs = GetModelBase(self.config)
        category_path = Dense(self.config.model.dense_units, activation='relu')(x)
        category_path = Dense(self.config.model.dense_units, activation='relu')(category_path)
        energy_path = Dense(self.config.model.dense_units, activation='relu')(x)
        energy_path = Dense(self.config.model.dense_units, activation='relu')(energy_path)
        category_output = Dense(self.categories, activation='softmax', name=self.cat)(category_path)
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
            't_nuEnergy': 0.0000005
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

        self.labels = ["EL-CC-QEL", "EL-CC-RES", "EL-CC-DIS", "EL-CC-COH",
                       "MU-CC-QEL", "MU-CC-RES", "MU-CC-DIS", "MU-CC-COH",
                       "EL-NC-QEL", "EL-NC-RES", "EL-NC-DIS", "EL-NC-COH",
                       "MU-NC-QEL", "MU-NC-RES", "MU-NC-DIS", "MU-NC-COH"]

    def combine_outputs(self, ev):
        """
        Combine outputs into fully combined categories.

        Args:
            ev (dict): Pandas single event(row) dict
        Returns:
            List[float, float, float]: List of combined category outputs
        """
        nuel = (ev['b_out_0'] + ev['b_out_1'] + ev['b_out_2'] + ev['b_out_3'])
        numu = (ev['b_out_4'] + ev['b_out_5'] + ev['b_out_6'] + ev['b_out_7'])
        nc = (ev['b_out_8'] + ev['b_out_9'] + ev['b_out_10'] + ev['b_out_11'] +
              ev['b_out_12'] + ev['b_out_13'] + ev['b_out_14'] + ev['b_out_15'])
        return [nuel, numu, nc]


class BeamAllInceptionModel(BaseModel):

    """
    Beam all category classification inception model class.
    """

    def __init__(self, config):
        """
        Initialise the BeamAllInceptionModel.

        Args:
            config (str): Dotmap configuration namespace
        """
        super().__init__(config)

    def build(self):
        """
        Builds the model using the keras functional API.
        """
        self.categories = 16
        self.cat = 't_cat'
        x, inputs = InceptionBase(self.config)
        outputs = Dense(self.categories, activation='softmax', name=self.cat)(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='beam_all_inception_model')
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['t_cat']

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           metrics=self.metrics)

        self.labels = ["EL-CC-QEL", "EL-CC-RES", "EL-CC-DIS", "EL-CC-COH",
                       "MU-CC-QEL", "MU-CC-RES", "MU-CC-DIS", "MU-CC-COH",
                       "EL-NC-QEL", "EL-NC-RES", "EL-NC-DIS", "EL-NC-COH",
                       "MU-NC-QEL", "MU-NC-RES", "MU-NC-DIS", "MU-NC-COH"]

    def combine_outputs(self, ev):
        """
        Combine outputs into fully combined categories.

        Args:
            ev (dict): Pandas single event(row) dict
        Returns:
            List[float, float, float]: List of combined category outputs
        """
        nuel = (ev['b_out_0'] + ev['b_out_1'] + ev['b_out_2'] + ev['b_out_3'])
        numu = (ev['b_out_4'] + ev['b_out_5'] + ev['b_out_6'] + ev['b_out_7'])
        nc = (ev['b_out_8'] + ev['b_out_9'] + ev['b_out_10'] + ev['b_out_11'] +
              ev['b_out_12'] + ev['b_out_13'] + ev['b_out_14'] + ev['b_out_15'])
        return [nuel, numu, nc]
