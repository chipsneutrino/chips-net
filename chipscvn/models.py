# -*- coding: utf-8 -*-

"""
tf.keras Functional API Model Implementations

This module contains the keras functional model definitions. All
are derived from the BaseModel class for ease-of-use in other parts
of the chips-cvn code
"""

import os
import tensorflow.keras.layers as layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow as tf

import chipscvn.blocks as blocks


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
        policy = mixed_precision.Policy(self.config.model.policy)
        mixed_precision.set_policy(policy)
        inputs, x = blocks.get_vgg16_base(self.config)
        x = layers.Dense(1, name='dense_logits')(x)
        outputs = layers.Activation('linear', dtype='float32', name=self.config.model.parameter)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='parameter_model')

        self.loss = 'mean_squared_error'
        self.metrics = ['mae', 'mse']
        self.es_monitor = 'val_mae'
        self.parameters = [self.config.model.parameter]

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.lr),
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
        policy = mixed_precision.Policy(self.config.model.policy)
        mixed_precision.set_policy(policy)
        inputs, x = blocks.get_vgg16_base(self.config)
        x = layers.Dense(1, name='dense_logits')(x)
        outputs = layers.Activation('sigmoid', dtype='float32', name='t_cosmic_cat')(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='cosmic_model')

        self.loss = 'binary_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['t_cosmic_cat']

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.lr),
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
        policy = mixed_precision.Policy(self.config.model.policy)
        mixed_precision.set_policy(policy)

        inputs, x = blocks.get_vgg16_base(self.config)
        x = layers.Dense(self.categories, name='dense_logits')(x)
        outputs = layers.Activation('softmax', dtype='float32', name=self.cat)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='beam_all_model')
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['t_cat']

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.lr),
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
        policy = mixed_precision.Policy(self.config.model.policy)
        mixed_precision.set_policy(policy)

        inputs, x = blocks.get_vgg16_base(self.config)
        x = layers.Dense(self.categories, name='dense_logits')(x)
        outputs = layers.Activation('softmax', dtype='float32', name=self.cat)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='beam_full_comb_model')
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['t_full_cat']

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.lr),
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
        policy = mixed_precision.Policy(self.config.model.policy)
        mixed_precision.set_policy(policy)

        inputs, x = blocks.get_vgg16_base(self.config)
        x = layers.Dense(self.categories, name='dense_logits')(x)
        outputs = layers.Activation('softmax', dtype='float32', name=self.cat)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='beam_nu_nc_comb_model')
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['t_nu_nc_cat']

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.lr),
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
        policy = mixed_precision.Policy(self.config.model.policy)
        mixed_precision.set_policy(policy)

        inputs, x = blocks.get_vgg16_base(self.config)
        x = layers.Dense(self.categories, name='dense_logits')(x)
        outputs = layers.Activation('softmax', dtype='float32', name=self.cat)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='beam_nc_comb_model')
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['t_nc_cat']

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.lr),
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
        policy = mixed_precision.Policy(self.config.model.policy)
        mixed_precision.set_policy(policy)

        inputs, x = blocks.get_vgg16_base(self.config)
        c_path = layers.Dense(self.config.model.dense_units, activation='relu')(x)
        c_path = layers.Dense(self.config.model.dense_units, activation='relu')(c_path)
        e_path = layers.Dense(self.config.model.dense_units, activation='relu')(x)
        e_path = layers.Dense(self.config.model.dense_units, activation='relu')(e_path)
        c_path = layers.Dense(self.categories, name='dense_logits')(c_path)
        c_out = layers.Activation('softmax', dtype='float32', name=self.cat)(c_path)
        e_path = layers.Dense(1, name='dense_logits')(e_path)
        e_out = layers.Activation('linear', dtype='float32', name='t_nuEnergy')(e_path)
        self.model = tf.keras.Model(inputs=inputs, outputs=[c_out, e_out], name='beam_multi_model')

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

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.lr),
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
        policy = mixed_precision.Policy(self.config.model.policy)
        mixed_precision.set_policy(policy)

        inputs, x = blocks.get_inceptionv1_base(self.config)
        outputs = layers.Activation('softmax', dtype='float32', name=self.cat)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='beam_nc_comb_model')
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='beam_all_inception_model')
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['t_cat']

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.lr),
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
