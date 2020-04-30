# -*- coding: utf-8 -*-

"""tf.keras Functional API Model Implementations

This module contains the keras functional model definitions. All
are derived from the BaseModel class for ease-of-use in other parts
of the chips-cvn code
"""

import os

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import chipscvn.data
import chipscvn.layers


def get_model(config):
    """Returns the correct model for the configuration.
    Args:
        config (dotmap.DotMap): DotMap Configuration namespace
    Returns:
        chipscvn.models model: Model class
    """
    if config.model.name == "parameter":
        return ParameterModel(config)
    elif config.model.name == "cosmic":
        return CosmicModel(config)
    elif config.model.name == "beam":
        return BeamModel(config)
    elif config.model.name == "multi_simple":
        return BeamMultiSimpleModel(config)
    elif config.model.name == "multi":
        return BeamMultiModel(config)
    else:
        raise NotImplementedError


class BaseModel:
    """Base model class which all model implementations derive from.
    """
    def __init__(self, config):
        """Initialise the BaseModel.
        Args:
            config (str): Dotmap configuration namespace
        """
        self.config = config
        self.build()

    def load(self, checkpoint_num=None):
        """Returns the correct model with its trained weights loaded.
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

    def summarise(self, model):
        """Prints model summary to stdout and plots diagram of model to file.
        Args:
            model (tf model): Model to summarise
        """
        model.summary()  # Print the model structure to stdout

        # Plot an image of the model to file
        file_name = os.path.join(self.config.exp.exp_dir, 'model_diagram.png')
        tf.keras.utils.plot_model(
            model,
            to_file=file_name,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=96
        )

    def build(self):
        """Build the model, overide in derived model class.
        """
        raise NotImplementedError


class ParameterModel(BaseModel):
    """Single parameter estimation model class.
    """
    def __init__(self, config):
        """Initialise the ParameterModel.
        Args:
            config (str): Dotmap configuration namespace
        """
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API.
        """
        policy = mixed_precision.Policy(self.config.model.precision_policy)
        mixed_precision.set_policy(policy)
        inputs, x = chipscvn.layers.get_vgg16_base(self.config)
        x = tf.keras.layers.Dense(1, name='dense_logits')(x)
        outputs = tf.keras.layers.Activation('linear', dtype='float32',
                                             name=self.config.model.labels[0])(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.config.model.name)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.lr),
                           loss='mean_squared_error', metrics=['mae', 'mse'])

        self.es_monitor = 'val_mae'
        self.parameters = [self.config.model.parameter]
        self.summarise(self.model)


class CosmicModel(BaseModel):
    """Cosmic vs beam classification model class.
    """
    def __init__(self, config):
        """Initialise the CosmicModel.
        Args:
            config (str): Dotmap configuration namespace
        """
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API.
        """
        policy = mixed_precision.Policy(self.config.model.precision_policy)
        mixed_precision.set_policy(policy)
        inputs, x = chipscvn.layers.get_vgg16_base(self.config)
        x = tf.keras.layers.Dense(1, name='dense_logits')(x)
        outputs = tf.keras.layers.Activation('sigmoid', dtype='float32',
                                             name=self.config.model.labels[0])(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.config.model.name)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.lr),
                           loss='binary_crossentropy', metrics=['accuracy'])

        self.es_monitor = 'val_accuracy'
        self.parameters = ['t_cosmic_cat']
        self.summarise(self.model)


class BeamModel(BaseModel):
    """Beam category classification model class.
    """
    def __init__(self, config):
        """Initialise the BeamModel.
        Args:
            config (str): Dotmap configuration namespace
        """
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API.
        """
        policy = mixed_precision.Policy(self.config.model.precision_policy)
        mixed_precision.set_policy(policy)
        inputs, x = chipscvn.layers.get_vgg16_base(self.config)
        x = tf.keras.layers.Dense(chipscvn.data.get_map(self.config.model.labels[0]).train_num,
                                  name='dense_logits')(x)
        outputs = tf.keras.layers.Activation('softmax', dtype='float32',
                                             name=self.config.model.labels[0])(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.config.model.name)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.lr),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.es_monitor = 'val_accuracy'
        self.parameters = [self.config.model.category]
        self.summarise(self.model)


class BeamMultiSimpleModel(BaseModel):
    """Simple Beam Multi-task model class.
    """
    def __init__(self, config):
        """Initialise the BeamMultiSimpleModel.
        Args:
            config (str): Dotmap configuration namespace
        """
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API.
        """
        policy = mixed_precision.Policy(self.config.model.precision_policy)
        mixed_precision.set_policy(policy)

        inputs, x = chipscvn.layers.get_vgg16_base(self.config)
        out_c = tf.keras.layers.Dense(self.config.model.dense_units, activation='relu')(x)
        out_c = tf.keras.layers.Dense(chipscvn.data.get_map(self.config.model.labels[0]).train_num,
                                      name='c_logits')(out_c)
        out_c = tf.keras.layers.Activation('softmax', dtype='float32',
                                           name=self.config.model.labels[0])(out_c)
        out_e = tf.keras.layers.Dense(self.config.model.dense_units, activation='relu')(x)
        out_e = tf.keras.layers.Dense(1, name='e_logits')(out_e)
        out_e = tf.keras.layers.Activation('linear', dtype='float32',
                                           name=self.config.model.labels[1])(out_e)

        self.model = tf.keras.Model(inputs=inputs, outputs=[out_c, out_e],
                                    name=self.config.model.name)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.lr),
            loss={self.config.model.category: 'sparse_categorical_crossentropy',
                  't_nuEnergy': 'mean_squared_error'},
            loss_weights={self.config.model.category: 1.0, 't_nuEnergy': 0.0000005},
            metrics={self.config.model.category: 'accuracy', 't_nuEnergy': 'mae'}
        )
        self.es_monitor = 'val_accuracy'
        self.parameters = [self.config.model.category, 't_nuEnergy']
        self.summarise(self.model)


class BeamMultiModel(BaseModel):
    """Beam Multi-task model class.
    """
    def __init__(self, config):
        """Initialise the BeamMultiModel.
        Args:
            config (str): Dotmap configuration namespace
        """
        super().__init__(config)

    def build(self):
        """Builds the model using the subclassing API
        """
        self.model = chipscvn.layers.CHIPSMultitask(
            self.config,
            chipscvn.data.get_map(self.config.model.labels[0]).train_num,
            self.config.model.labels[0]
        )
        input_shape = [[self.config.data.batch_size] + self.config.data.img_size, 1, 1]
        print(input_shape)
        self.model.build(input_shape)
        self.summarise(self.model.model())
