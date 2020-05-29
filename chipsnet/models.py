# -*- coding: utf-8 -*-

"""tf.keras Functional API Model Implementations

This module contains the keras functional model definitions. All
are derived from the BaseModel class for ease-of-use in other parts
of the chipsnet code
"""

import os

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import chipsnet.data as data
import chipsnet.layers


def get_model(config):
    """Returns the correct model for the configuration.
    Args:
        config (dotmap.DotMap): DotMap Configuration namespace
    Returns:
        chipsnet.models model: Model class
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
                self.config.exp.checkpoints_dir + "cp-" + str(checkpoint_num).zfill(4) + ".ckpt")
            self.model.load_weights(checkpoint_path).expect_partial()

    def summarise(self, model):
        """Prints model summary to stdout and plots diagram of model to file.
        Args:
            model (tf model): Model to summarise
        """
        model.summary()  # Print the model structure to stdout

        # Plot an image of the model to file
        file_name = os.path.join(self.config.exp.exp_dir, "model_diagram.png")
        tf.keras.utils.plot_model(
            model,
            to_file=file_name,
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
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
        inputs, x = chipsnet.layers.get_vgg16_base(self.config)
        x = tf.keras.layers.Dense(1, name="dense_logits")(x)
        outputs = tf.keras.layers.Activation("linear", dtype="float32",
                                             name=self.config.model.labels[0])(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.config.model.name)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.lr),
                           loss="mean_squared_error", metrics=["mae", "mse"])

        self.es_monitor = "val_mae"
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
        inputs, x = chipsnet.layers.get_vgg16_base(self.config)
        x = tf.keras.layers.Dense(1, name="dense_logits")(x)
        outputs = tf.keras.layers.Activation("sigmoid", dtype="float32",
                                             name=self.config.model.labels[0])(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.config.model.name)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.lr),
                           loss="binary_crossentropy", metrics=["accuracy"])

        self.es_monitor = "val_accuracy"
        self.parameters = ["t_cosmic_cat"]
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
        inputs, x = chipsnet.layers.get_vgg16_base(self.config)
        x = tf.keras.layers.Dense(data.get_map(
            self.config.model.labels[0]).categories, name="dense_logits")(x)
        outputs = tf.keras.layers.Activation("softmax", dtype="float32",
                                             name=self.config.model.labels[0])(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.config.model.name)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.lr),
                           loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        self.es_monitor = "val_accuracy"
        self.parameters = [self.config.model.labels[0]]
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

        # Get the base of the model...
        inputs, x = chipsnet.layers.get_vgg16_base(self.config)

        outputs = []
        losses, weights, metrics = {}, {}, {}
        for output in self.config.model.labels:

            if output == data.MAP_NU_TYPE.name:
                out = tf.keras.layers.Dense(
                    self.config.model.dense_units,
                    activation="relu",
                    name=output+"_dense")(x)
                out = tf.keras.layers.Dense(
                    1,
                    name=output+"_logits")(out)
                outputs.append(tf.keras.layers.Activation(
                    "softmax",
                    dtype="float32",
                    name=output)(out))
                losses[output] = "sigmoid"
                weights[output] = 1.0
                metrics[output] = "accuracy"

            elif output == data.MAP_SIGN_TYPE.name:
                out = tf.keras.layers.Dense(
                    self.config.model.dense_units,
                    activation="relu",
                    name=output+"_dense")(x)
                out = tf.keras.layers.Dense(
                    1,
                    name=output+"_logits")(out)
                outputs.append(tf.keras.layers.Activation(
                    "softmax",
                    dtype="float32",
                    name=output)(out))
                losses[output] = "sigmoid"
                weights[output] = 1.0
                metrics[output] = "accuracy"

            elif output == data.MAP_INT_TYPE.name:
                out = tf.keras.layers.Dense(
                    self.config.model.dense_units,
                    activation="relu",
                    name=output+"_dense")(x)
                out = tf.keras.layers.Dense(
                    data.get_map(output).categories,
                    name=output+"_logits")(out)
                outputs.append(tf.keras.layers.Activation(
                    "softmax",
                    dtype="float32",
                    name=output)(out))
                losses[output] = "sparse_categorical_crossentropy"
                weights[output] = 1.0
                metrics[output] = "accuracy"

            elif output == data.MAP_ALL_CAT.name:
                out = tf.keras.layers.Dense(
                    self.config.model.dense_units,
                    activation="relu",
                    name=output+"_dense")(x)
                out = tf.keras.layers.Dense(
                    data.get_map(output).categories,
                    name=output+"_logits")(out)
                outputs.append(tf.keras.layers.Activation(
                    "softmax",
                    dtype="float32",
                    name=output)(out))
                losses[output] = "sparse_categorical_crossentropy"
                weights[output] = 1.0
                metrics[output] = "accuracy"

            elif output == data.MAP_COSMIC_CAT.name:
                out = tf.keras.layers.Dense(
                    self.config.model.dense_units,
                    activation="relu",
                    name=output+"_dense")(x)
                out = tf.keras.layers.Dense(
                    1,
                    name=output+"_logits")(out)
                outputs.append(tf.keras.layers.Activation(
                    "softmax",
                    dtype="float32",
                    name=output)(out))
                losses[output] = "sigmoid"
                weights[output] = 1.0
                metrics[output] = "accuracy"

            elif output == data.MAP_FULL_COMB_CAT.name:
                out = tf.keras.layers.Dense(
                    self.config.model.dense_units,
                    activation="relu",
                    name=output+"_dense")(x)
                out = tf.keras.layers.Dense(
                    data.get_map(output).categories,
                    name=output+"_logits")(out)
                outputs.append(tf.keras.layers.Activation(
                    "softmax",
                    dtype="float32",
                    name=output)(out))
                losses[output] = "sparse_categorical_crossentropy"
                weights[output] = 1.0
                metrics[output] = "accuracy"

            elif output == data.MAP_NU_NC_COMB_CAT.name:
                out = tf.keras.layers.Dense(
                    self.config.model.dense_units,
                    activation="relu",
                    name=output+"_dense")(x)
                out = tf.keras.layers.Dense(
                    data.get_map(output).categories,
                    name=output+"_logits")(out)
                outputs.append(tf.keras.layers.Activation(
                    "softmax",
                    dtype="float32",
                    name=output)(out))
                losses[output] = "sparse_categorical_crossentropy"
                weights[output] = 1.0
                metrics[output] = "accuracy"

            elif output == data.MAP_NC_COMB_CAT.name:
                out = tf.keras.layers.Dense(
                    self.config.model.dense_units,
                    activation="relu",
                    name=output+"_dense")(x)
                out = tf.keras.layers.Dense(
                    data.get_map(output).categories,
                    name=output+"_logits")(out)
                outputs.append(tf.keras.layers.Activation(
                    "softmax",
                    dtype="float32",
                    name=output)(out))
                losses[output] = "sparse_categorical_crossentropy"
                weights[output] = 1.0
                metrics[output] = "accuracy"

            elif output == "t_nuEnergy":
                out = tf.keras.layers.Dense(
                    self.config.model.dense_units,
                    activation="relu",
                    name=output+"_dense")(x)
                out = tf.keras.layers.Dense(
                    1,
                    name=output+"_logits")(out)
                outputs.append(tf.keras.layers.Activation(
                    "linear",
                    dtype="float32",
                    name=output)(out))
                losses[output] = "mean_squared_error"
                weights[output] = 0.0000005
                metrics[output] = "mae"

            elif output in ["t_vtxX", "t_vtxY", "t_vtxZ"]:
                out = tf.keras.layers.Dense(
                    self.config.model.dense_units,
                    activation="relu",
                    name=output+"_dense")(x)
                out = tf.keras.layers.Dense(
                    1,
                    name=output+"_logits")(out)
                outputs.append(tf.keras.layers.Activation(
                    "linear",
                    dtype="float32",
                    name=output)(out))
                losses[output] = "mean_squared_error"
                weights[output] = 0.00001
                metrics[output] = "mae"

            if output in ["prim_total", "prim_p", "prim_cp", "prim_np", "prim_g"]:
                out = tf.keras.layers.Dense(
                    self.config.model.dense_units,
                    activation="relu",
                    name=output+"_dense")(x)
                out = tf.keras.layers.Dense(
                    4,
                    name=output+"_logits")(out)
                outputs.append(tf.keras.layers.Activation(
                    "softmax",
                    dtype="float32",
                    name=output)(out))
                losses[output] = "sparse_categorical_crossentropy"
                weights[output] = 1.0
                metrics[output] = "accuracy"

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.config.model.name)
        optimiser = tf.keras.optimizers.Adam(learning_rate=self.config.model.lr)
        self.model.compile(optimizer=optimiser, loss=losses, loss_weights=weights, metrics=metrics)
        self.es_monitor = self.config.model.labels[0]  # We monitor the first output in the list
        self.parameters = [self.config.model.labels[0], self.config.model.labels[1]]
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
        self.model = chipsnet.layers.CHIPSMultitask(
            self.config,
            data.get_map(self.config.model.labels[0]).categories,
            self.config.model.labels[0]
        )
        self.summarise(self.model.model())
