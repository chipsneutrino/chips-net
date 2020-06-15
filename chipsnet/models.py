# -*- coding: utf-8 -*-

"""tf.keras Functional API Model Implementations

This module contains the keras functional model definitions. All
are derived from the BaseModel class for ease-of-use in other parts
of the chipsnet code

An illustrated guide to many of the blocks is at...
https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d
"""

import os

import tensorflow as tf
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.initializers as initializers
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Dropout,
    GlobalAveragePooling2D,
    Reshape,
    add,
    Dense,
    multiply,
    concatenate,
    Flatten,
    AveragePooling2D,
    Lambda,
    Concatenate,
)
from tensorflow.keras.losses import (
    SparseCategoricalCrossentropy,
    MeanSquaredError,
    BinaryCrossentropy,
    Reduction,
)

import chipsnet.data as data


class Model:
    """chipsnet model class containing many useful management methods.
    """

    def __init__(self, config):
        """Initialise the chipsnet model class.
        Args:
            config (str): Dotmap configuration namespace
        """
        self.config = config
        self.build()  # We build the model immediately

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
                self.config.exp.checkpoints_dir
                + "cp-"
                + str(checkpoint_num).zfill(4)
                + ".ckpt"
            )
            self.model.load_weights(checkpoint_path).expect_partial()

    def summarise(self):
        """Prints model summary to stdout and plots diagram of model to file.
        Args:
            model (tf model): Model to summarise
        """
        self.model.summary()  # Print the model structure to stdout

        # Plot an image of the model to file
        file_name = os.path.join(self.config.exp.exp_dir, "model_diagram.png")
        self.model._layers = [
            layer
            for layer in self.model._layers
            if isinstance(layer, tf.keras.layers.Layer)
        ]
        tf.keras.utils.plot_model(
            self.model,
            to_file=file_name,
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )

    def build(self):
        """Build the model.
        """
        policy = mixed_precision.Policy(self.config.model.precision_policy)
        mixed_precision.set_policy(policy)
        if self.config.model.type == "vgg":
            self.model = vgg_model(self.config)
        elif self.config.model.type == "resnet":
            self.model = resnet_model(self.config)
        elif self.config.model.type == "inception":
            self.model = inception_resnet_model(self.config)
        else:
            raise NotImplementedError

        if self.config.model.summarise:
            self.summarise()


CONV_INITIALISER = "glorot_uniform"  # he_normal, glorot_uniform
DENSE_INITIALISER = "glorot_uniform"  # he_normal, glorot_uniform


def conv2d_bn(
    x,
    filters,
    kernel_size,
    strides=1,
    padding="same",
    activation="relu",
    use_bias=False,
    prefix=None,
):
    """Utility function to apply conv + BN.
    Args:
        x (tf.tensor): input tensor.
        filters (int): Number of filters in the Conv2D
        kernel_size (int): Kernel size in the Conv2D
        strides (int): Strides in the Conv2D
        padding (str): Padding mode in Conv2D
        activation (str): Activation in Conv2D
        use_bias (bool): Whether to use a bias in Conv2D
        prefix (str): Prefix to prepend to all layer names
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=CONV_INITIALISER,
        name=prefix,
    )(x)
    if not use_bias:
        bn_name = None if prefix is None else prefix + "_bn"
        x = BatchNormalization(axis=3, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if prefix is None else prefix + "_ac"
        x = Activation(activation, name=ac_name)(x)
    return x


def vgg_block(x, num_conv=2, filters=64, se_ratio=0, dropout=0.0, prefix=""):
    """Builds a VGG block.
    This function builds a vgg block as defined by the 'Visual Geometry Group' at Oxford and
    layed out in the paper https://arxiv.org/abs/1409.1556. We additionally add the now prevalent
    batch normalisation after the convolutional layer. Optionally, a squeeze-exitation block
    as first set out in https://arxiv.org/abs/1709.01507 can be added, as well as a dropout layer.
    The following was used to guide the implementation,
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py.
    Args:
        x (tf.tensor): Input tf tensor
        num_conv (int): Number of convolutional layers
        filters (int): Number of filters to use in convolutional layers
        se_ratio (int): Squeeze-exitation ratio if '0' will not be used
        dropout (float): What should the dropout rate be?
        prefix (str): Prefix to prepend to all layer names
    Returns:
        tf.tensor: Output tf tensor from the vgg block
    Raises:
        ValueError: if 'num_conv' is not one of 2, 3, or 4.
    """
    conv_options = [2, 3, 4]
    if num_conv not in conv_options:
        raise ValueError("Invalid num_conv. Expected one of: {}".format(conv_options))

    for i in range(num_conv):
        x = conv2d_bn(x, filters, (3, 3), prefix=prefix + "_conv" + str(i))
    x = MaxPooling2D((2, 2), strides=(2, 2), name=prefix + "_pool")(x)

    x = squeeze_excite_block(x, se_ratio, prefix=prefix) if se_ratio > 0 else x
    x = Dropout(dropout, name=prefix + "_drop")(x) if dropout > 0.0 else x
    return x


def resnet_block(
    x,
    filters,
    k=1,
    strides=(1, 1),
    bottleneck=False,
    se_ratio=0,
    dropout=0.0,
    prefix="",
):
    """Builds a resnet block
    This function builds pre-activation resnet block using the improved structure layed
    out in https://arxiv.org/abs/1603.05027. Optionally, a squeeze-exitation block as first set
    out in https://arxiv.org/abs/1709.01507 can be added, as well as a dropout layer. The option
    of using the bottleneck implementation instead is also an option. The following proved
    helpful in the implementation...
        - https://github.com/titu1994/keras-squeeze-excite-network/blob/master/ \
            keras_squeeze_excite_network/se_resnet.py
        - https://github.com/keras-team/keras-applications/blob/master/ \
            keras_applications/resnet_common.py
        - https://github.com/kobiso/SENet-tensorflow-slim/blob/master/nets/resnet_v2.py
    Args:
        x (tf.tensor): Input tf tensor
        filters (int): Number of output filters
        k (int): Width factor
        strides (int): Strides of the convolution layer
        bottleneck (bool): Should we use the bottleneck variant?
        se_ratio (int): Squeeze-exitation ratio if '0' will not be used
        dropout (float): What should the dropout rate be?
        prefix (str): Prefix to prepend to all layer names
    Returns:
        tf.tensor: Output tf tensor from the resnet block
    """
    init = x

    # Add the preactivation layers
    x = BatchNormalization(axis=3, name=prefix + "_preact_bn")(x)
    x = Activation("relu", name=prefix + "_preact_ac")(x)

    # Apply a convolution to the shortcut if required
    shortcut_f = filters * k * 4 if bottleneck else filters * k
    if strides != (1, 1) or init.shape[3] != shortcut_f:
        init = Conv2D(
            shortcut_f,
            (1, 1),
            padding="same",
            kernel_initializer=CONV_INITIALISER,
            use_bias=False,
            strides=strides,
            name=prefix + "_conv0",
        )(x)

    if bottleneck:
        x = Conv2D(
            filters * k,
            (1, 1),
            padding="same",
            kernel_initializer=CONV_INITIALISER,
            use_bias=False,
            name=prefix + "_conv1",
        )(x)
        x = BatchNormalization(axis=3, name=prefix + "_bn1")(x)
        x = Activation("relu", name=prefix + "_ac1")(x)

        x = Conv2D(
            filters * k,
            (3, 3),
            padding="same",
            kernel_initializer=CONV_INITIALISER,
            use_bias=False,
            strides=strides,
            name=prefix + "_conv2",
        )(x)
        x = BatchNormalization(axis=3, name=prefix + "_bn2")(x)
        x = Activation("relu", name=prefix + "_ac2")(x)

        x = Conv2D(
            shortcut_f,
            (1, 1),
            padding="same",
            kernel_initializer=CONV_INITIALISER,
            use_bias=False,
            name=prefix + "_conv3",
        )(x)
    else:
        x = Conv2D(
            filters * k,
            (3, 3),
            padding="same",
            kernel_initializer=CONV_INITIALISER,
            use_bias=False,
            strides=strides,
            name=prefix + "_conv1",
        )(x)
        x = BatchNormalization(axis=3, name=prefix + "_bn1")(x)
        x = Activation("relu", name=prefix + "_ac1")(x)

        x = Conv2D(
            filters * k,
            (3, 3),
            padding="same",
            kernel_initializer=CONV_INITIALISER,
            use_bias=False,
            name=prefix + "_conv2",
        )(x)

    x = squeeze_excite_block(x, se_ratio, prefix=prefix) if se_ratio > 0 else x
    x = Dropout(dropout, name=prefix + "_drop")(x) if dropout > 0.0 else x
    x = add([x, init])  # Add the residual and shortcut together
    return x


def inception_resnet_block(
    x, scale, block_type, activation="relu", se_ratio=0, dropout=0.0, prefix=""
):
    """Builds an Inception-ResNet block
    This function builds 3 types of Inception-ResNet blocks mentioned in the paper
    https://arxiv.org/abs/1602.07261, controlled by the block_type argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: block_type='block35'
        - Inception-ResNet-B: block_type='block17'
        - Inception-ResNet-C: block_type='block8'
    Implementation taken from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/ \
    python/keras/applications/inception_resnet_v2.py. Optionally, a squeeze-exitation block as
    first set out in https://arxiv.org/abs/1709.01507 can be added, as well as a dropout layer
    Args:
        x (tf.tensor): Input tf tensor
        scale (float): scaling factor to scale the residuals
        block_type (str): 'block35', 'block17' or 'block8'
        activation (str): Activation function to use at the end of the block
        se_ratio (int): Squeeze-exitation ratio if '0' will not be used
        dropout (float): What should the dropout rate be?
        prefix (str): Prefix to prepend to all layer names
    Returns:
        Output tensor for the inception resnet block.
    Raises:
        ValueError: if 'block_type' is not one of 'block35', 'block17' or 'block8'.
    """
    if block_type == "block35":
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == "block17":
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == "block8":
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError(
            "Unknown Inception-ResNet block type. "
            'Expects "block35", "block17" or "block8", '
            "but got: {}".format(block_type)
        )

    mixed = Concatenate(axis=3, name=prefix + "_mixed")(branches)
    up = conv2d_bn(
        mixed,
        tf.keras.backend.int_shape(x)[3],
        1,
        activation=None,
        use_bias=True,
        prefix=prefix + "_conv",
    )

    x = Lambda(
        lambda inputs, scale: inputs[0] + inputs[1] * scale,
        output_shape=tf.keras.backend.int_shape(x)[1:],
        arguments={"scale": scale},
        name=prefix,
    )([x, up])
    if activation is not None:
        x = Activation(activation, name=prefix + "_ac")(x)

    x = squeeze_excite_block(x, se_ratio, prefix=prefix) if se_ratio > 0 else x
    x = Dropout(dropout, name=prefix + "_drop")(x) if dropout > 0.0 else x
    return x


def squeeze_excite_block(x, ratio=16, prefix=""):
    """Builds a squeeze-exitation block
    This function builds a squeeze-exitation block as first set out in
    https://arxiv.org/abs/1709.01507. The implementation used here was taken from
    https://github.com/titu1994/keras-squeeze-excite-network/blob/master/ \
        keras_squeeze_excite_network/se.py
    Args:
        x (tf.tensor): Input tf tensor
        ratio (int): Squeeze ratio
        prefix (str): Prefix to prepend to all layer names
    Returns:
        Output tensor for the squeeze_excite_block block.
    """

    init = x
    filters = init.shape[3]
    se_shape = (1, 1, filters)
    se = GlobalAveragePooling2D(name=prefix + "_se_pool")(init)
    se = Reshape(se_shape, name=prefix + "_se_reshape")(se)
    se = Dense(
        filters // ratio,
        activation="relu",
        kernel_initializer=DENSE_INITIALISER,
        use_bias=False,
        name=prefix + "_se_squeeze",
    )(se)
    se = Dense(
        filters,
        activation="sigmoid",
        kernel_initializer=DENSE_INITIALISER,
        use_bias=False,
        name=prefix + "_se_excite",
    )(se)
    x = multiply([init, se], name=prefix + "_se_multiply")
    return x


def vgg_model(config):
    """Builds the VGG-16 model
    Args:
        config (dotmap): Dotmap configuration namespace
    Returns:
        tf.keras.Model: VGG keras model
    """

    # Get the image inputs
    inputs = []
    inputs.extend(get_image_inputs(config))

    # Build the core of the model
    paths = []
    for i, image_input in enumerate(inputs):
        path = vgg_block(
            image_input,
            2,
            config.model.filters,
            config.model.se_ratio,
            config.model.dropout,
            prefix="block1_path" + str(i),
        )
        paths.append(path)

    x = paths[0]
    if len(paths) > 1:
        x = concatenate(paths)

    x = vgg_block(
        x,
        2,
        config.model.filters * 2,
        config.model.se_ratio,
        config.model.dropout,
        prefix="block2",
    )
    x = vgg_block(
        x,
        3,
        config.model.filters * 4,
        config.model.se_ratio,
        config.model.dropout,
        prefix="block3",
    )
    x = vgg_block(
        x,
        3,
        config.model.filters * 8,
        config.model.se_ratio,
        config.model.dropout,
        prefix="block4",
    )
    x = vgg_block(
        x,
        3,
        config.model.filters * 8,
        config.model.se_ratio,
        config.model.dropout,
        prefix="block5",
    )
    x = Flatten(name="flatten")(x)

    # Add the reco parameters as inputs if required
    if config.model.reco_pars:
        reco_inputs = get_reco_inputs()
        inputs.extend(reco_inputs)
        reco_concat = concatenate(reco_inputs)
        x = concatenate([x, reco_concat])

    x = Dense(
        config.model.dense_units,
        activation="relu",
        kernel_initializer=DENSE_INITIALISER,
        name="dense1",
    )(x)
    x = Dense(
        config.model.dense_units,
        activation="relu",
        kernel_initializer=DENSE_INITIALISER,
        name="dense_final",
    )(x)
    x = Dropout(config.model.dropout, name="dropout_final")(x)

    # Get the model outputs
    outputs, lwm = get_outputs(config, x)

    # If we want to use the custom loss weight learning layer add now and then
    # compile and return the model
    model = None
    if config.model.learn_weights:
        label_inputs, outputs = add_multitask_loss(config, outputs)
        inputs.extend(label_inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=config.model.name)
        optimiser = tf.keras.optimizers.Adam(learning_rate=config.model.lr)
        model.compile(optimizer=optimiser, loss=None, loss_weights=None, metrics=lwm[2])
    else:
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=config.model.name)
        optimiser = tf.keras.optimizers.Adam(learning_rate=config.model.lr)
        model.compile(
            optimizer=optimiser, loss=lwm[0], loss_weights=lwm[1], metrics=lwm[2]
        )

    return model


def resnet_model(config):
    """Builds the resnet model
    Args:
        config (dotmap): Dotmap configuration namespace
    Returns:
        tf.keras.Model: Resnet keras model
    """

    depths = [3, 4, 6, 3]
    filters = [64, 128, 256, 512]

    # Get the image inputs
    inputs = []
    inputs.extend(get_image_inputs(config))

    # Build the core of the model
    paths = []
    for i, image_input in enumerate(inputs):
        path = Conv2D(
            filters[0],
            (3, 3),
            padding="same",
            use_bias=False,
            strides=(2, 2),
            kernel_initializer=CONV_INITIALISER,
            kernel_regularizer=regularizers.l2(1e-4),
            name="conv1_path" + str(i),
        )(image_input)
        # path = MaxPooling2D((3, 3), strides=(2, 2), padding='same',
        #                           name="pool1_path" + str(i))(path)
        for j in range(depths[0]):
            path = resnet_block(
                path,
                filters[0],
                k=1,
                strides=(1, 1),
                bottleneck=config.model.bottleneck,
                se_ratio=config.model.se_ratio,
                dropout=config.model.dropout,
                prefix="path" + str(i) + "_block0_" + str(j),
            )
        paths.append(path)

    x = paths[0]
    if len(paths) > 1:
        x = concatenate(paths)

    for k in range(1, len(depths)):
        x = resnet_block(
            x,
            filters[k],
            k=1,
            strides=(2, 2),
            bottleneck=config.model.bottleneck,
            se_ratio=config.model.se_ratio,
            dropout=config.model.dropout,
            prefix="block" + str(k) + "_0",
        )
        for i in range(depths[k] - 1):
            x = resnet_block(
                x,
                filters[k],
                k=1,
                strides=(1, 1),
                bottleneck=config.model.bottleneck,
                se_ratio=config.model.se_ratio,
                dropout=config.model.dropout,
                prefix="block" + str(k) + "_" + str(i + 1),
            )

    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(
        config.model.dense_units,
        activation="relu",
        kernel_initializer=DENSE_INITIALISER,
        name="dense_final",
    )(x)

    # Add the reco parameters as inputs if required
    if config.model.reco_pars:
        reco_inputs = get_reco_inputs()
        inputs.extend(reco_inputs)
        reco_concat = concatenate(reco_inputs)
        x = concatenate([x, reco_concat])

    x = Dropout(config.model.dropout, name="dropout_final")(x)

    # Get the model outputs
    outputs, lwm = get_outputs(config, x)

    # If we want to use the custom loss weight learning layer add now and then
    # compile and return the model
    model = None
    if config.model.learn_weights:
        label_inputs, outputs = add_multitask_loss(config, outputs)
        inputs.extend(label_inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=config.model.name)
        optimiser = tf.keras.optimizers.Adam(learning_rate=config.model.lr)
        model.compile(optimizer=optimiser, loss=None, loss_weights=None, metrics=lwm[2])
    else:
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=config.model.name)
        optimiser = tf.keras.optimizers.Adam(learning_rate=config.model.lr)
        model.compile(
            optimizer=optimiser, loss=lwm[0], loss_weights=lwm[1], metrics=lwm[2]
        )

    return model


def inception_resnet_model(config):
    """Builds the Inception-Resnet-v2 model
    Args:
        config (dotmap): Dotmap configuration namespace
    Returns:
        tf.keras.Model: Inception-Resnet-v2 keras model
    """

    blocks = [5, 5, 5]  # [11, 21, 10]
    scales = [0.17, 0.1, 0.2]

    # Get the image inputs
    inputs = []
    inputs.extend(get_image_inputs(config))

    # Build the core of the model
    paths = []
    for i, image_input in enumerate(inputs):
        # Stem block: 35 x 35 x 192
        path = conv2d_bn(image_input, 32, 3, padding="valid")
        path = conv2d_bn(path, 64, 3)
        path = MaxPooling2D(3, strides=2)(path)
        # original version
        # path = conv2d_bn(image_input, 32, 3, strides=2, padding='valid')
        # path = conv2d_bn(path, 32, 3, padding='valid')
        # path = conv2d_bn(path, 64, 3)
        # path = MaxPooling2D(3, strides=2)(path)
        # path = conv2d_bn(path, 80, 1, padding='valid')
        # path = conv2d_bn(path, 192, 3, padding='valid')
        # path = MaxPooling2D(3, strides=2)(path)
        paths.append(path)

    x = paths[0]
    if len(paths) > 1:
        x = concatenate(paths)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_pool = AveragePooling2D(3, strides=1, padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=3, name="mixed_5b")(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, blocks[0]):
        x = inception_resnet_block(
            x, scale=scales[0], block_type="block35", prefix="block35_" + str(block_idx)
        )

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding="valid")
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding="valid")
    branch_pool = MaxPooling2D(3, strides=2, padding="valid")(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=3, name="mixed_6a")(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, blocks[1]):
        x = inception_resnet_block(
            x, scale=scales[1], block_type="block17", prefix="block17_" + str(block_idx)
        )

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding="valid")
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding="valid")
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding="valid")
    branch_pool = MaxPooling2D(3, strides=2, padding="valid")(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=3, name="mixed_7a")(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, blocks[2]):
        x = inception_resnet_block(
            x, scale=scales[2], block_type="block8", prefix="block8_" + str(block_idx)
        )
    x = inception_resnet_block(
        x,
        scale=1.0,
        activation=None,
        block_type="block8",
        prefix="block8_" + str(blocks[2]),
    )

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, prefix="conv_7b")

    x = GlobalAveragePooling2D(name="avg_pool")(x)

    x = Dense(
        config.model.dense_units,
        activation="relu",
        kernel_initializer=DENSE_INITIALISER,
        name="dense1",
    )(x)
    x = Dense(
        config.model.dense_units,
        activation="relu",
        kernel_initializer=DENSE_INITIALISER,
        name="dense_final",
    )(x)
    x = Dropout(config.model.dropout, name="dropout_final")(x)

    # Get the model outputs
    outputs, lwm = get_outputs(config, x)

    # If we want to use the custom loss weight learning layer add now and then
    # compile and return the model
    model = None
    if config.model.learn_weights:
        label_inputs, outputs = add_multitask_loss(config, outputs)
        inputs.extend(label_inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=config.model.name)
        optimiser = tf.keras.optimizers.Adam(learning_rate=config.model.lr)
        model.compile(optimizer=optimiser, loss=None, loss_weights=None, metrics=lwm[2])
    else:
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=config.model.name)
        optimiser = tf.keras.optimizers.Adam(learning_rate=config.model.lr)
        model.compile(
            optimizer=optimiser, loss=lwm[0], loss_weights=lwm[1], metrics=lwm[2]
        )

    return model


def get_image_inputs(config):
    """Generates the image inputs for the model
    Args:
        config (dotmap): Dotmap configuration namespace
    Returns:
        list[tf.keras.Inputs]: Image inputs to the model
    """
    inputs = []
    images = 1
    shape = (
        config.data.img_size[0],
        config.data.img_size[1],
        config.data.channels.count(1),
    )
    if config.data.unstack:
        images = config.data.channels.count(1)
        shape = (config.data.img_size[0], config.data.img_size[1], 1)

    for channel in range(images):
        inputs.append(tf.keras.Input(shape=shape, name="image_" + str(channel)))

    return inputs


def get_reco_inputs():
    """Generates the reconstructed parameters inputs
    Args:
        config (dotmap): Dotmap configuration namespace
    Returns:
        list[tf.keras.Inputs]: Reco inputs to the model
    """
    inputs = []
    parameters = ["r_vtxX", "r_vtxY", "r_vtxZ", "r_dirTheta", "r_dirPhi"]
    for name in parameters:
        inputs.append(tf.keras.Input(shape=(1), name=name))
    return inputs


def get_outputs(config, x):
    """Generates the outputs to the model
    Args:
        config (dotmap): Dotmap configuration namespace
        x (tf.tensor): Input tensor of model
    Returns:
        list[tf.tensor]: Model outputs
        tuple(losses, weights, metrics): Tuple of model compilation arguments
    Raises:
        ValueError: if labels is not a list with length atleast 1
    """

    if type(config.model.labels) != list:
        raise ValueError("Invalid labels type, must be a list")
    elif len(config.model.labels) == 0:
        raise ValueError("Invalid labels length, must be greater than zero")

    outputs = []
    losses, weights, metrics = {}, {}, {}
    for output in config.model.labels:
        if output == data.MAP_NU_TYPE.name:
            out = Dense(1, name=output + "_logits")(x)
            outputs.append(Activation("sigmoid", dtype="float32", name=output)(out))
            losses[output] = "binary_crossentropy"
            weights[output] = 1.0
            metrics[output] = "accuracy"

        elif output == data.MAP_SIGN_TYPE.name:
            out = Dense(1, name=output + "_logits")(x)
            outputs.append(Activation("sigmoid", dtype="float32", name=output)(out))
            losses[output] = "binary_crossentropy"
            weights[output] = 1.0
            metrics[output] = "accuracy"

        elif output == data.MAP_INT_TYPE.name:
            out = Dense(data.get_map(output).categories, name=output + "_logits")(x)
            outputs.append(Activation("softmax", dtype="float32", name=output)(out))
            losses[output] = "sparse_categorical_crossentropy"
            weights[output] = 1.0
            metrics[output] = "accuracy"

        elif output == data.MAP_ALL_CAT.name:
            out = Dense(data.get_map(output).categories, name=output + "_logits")(x)
            outputs.append(Activation("softmax", dtype="float32", name=output)(out))
            losses[output] = "sparse_categorical_crossentropy"
            weights[output] = 1.0
            metrics[output] = "accuracy"

        elif output == data.MAP_COSMIC_CAT.name:
            out = Dense(1, name=output + "_logits")(x)
            outputs.append(Activation("sigmoid", dtype="float32", name=output)(out))
            losses[output] = "binary_crossentropy"
            weights[output] = 1.0
            metrics[output] = "accuracy"

        elif output == data.MAP_FULL_COMB_CAT.name:
            out = Dense(data.get_map(output).categories, name=output + "_logits")(x)
            outputs.append(Activation("softmax", dtype="float32", name=output)(out))
            losses[output] = "sparse_categorical_crossentropy"
            weights[output] = 1.0
            metrics[output] = "accuracy"

        elif output == data.MAP_NU_NC_COMB_CAT.name:
            out = Dense(data.get_map(output).categories, name=output + "_logits")(x)
            outputs.append(Activation("softmax", dtype="float32", name=output)(out))
            losses[output] = "sparse_categorical_crossentropy"
            weights[output] = 1.0
            metrics[output] = "accuracy"

        elif output == data.MAP_NC_COMB_CAT.name:
            out = Dense(data.get_map(output).categories, name=output + "_logits")(x)
            outputs.append(Activation("softmax", dtype="float32", name=output)(out))
            losses[output] = "sparse_categorical_crossentropy"
            weights[output] = 1.0
            metrics[output] = "accuracy"

        elif output in ["t_vtxX", "t_vtxY", "t_vtxZ"]:
            out = Dense(1, name=output + "_logits")(x)
            outputs.append(Activation("linear", dtype="float32", name=output)(out))
            losses[output] = "mean_squared_error"
            weights[output] = 0.000001
            metrics[output] = "mae"

        if output in ["prim_total", "prim_p", "prim_cp", "prim_np", "prim_g"]:
            out = Dense(4, name=output + "_logits")(x)
            outputs.append(Activation("softmax", dtype="float32", name=output)(out))
            losses[output] = "sparse_categorical_crossentropy"
            weights[output] = 1.0
            metrics[output] = "accuracy"

        elif output == "t_nuEnergy":
            out = Dense(config.model.dense_units, name="energy_dense")(x)
            out = Dense(1, name=output + "_logits")(out)
            outputs.append(Activation("linear", dtype="float32", name=output)(out))
            losses[output] = "mean_squared_error"
            weights[output] = 0.0000005
            metrics[output] = "mae"

    return outputs, (losses, weights, metrics)


def add_multitask_loss(config, outputs):
    """Add the multitask loss layer thats 'learns' how to weight the losses
    Args:
        config (dotmap): Dotmap configuration namespace
        outputs (tf.tensor): Model outputs
    Returns:
        tf.tensor: Additional inputs for the labels
        tf.tensor: Model outputs
    """
    inputs = []
    for label in config.model.labels:
        inputs.append(tf.keras.Input(shape=(1), name="input_" + label))
    outputs.append(MultiLossLayer(config, name="multiloss")(inputs + outputs))
    return inputs, outputs


class MultiLossLayer(tf.keras.layers.Layer):
    """Weighted multi-loss layer for multitask network
    A layer to calculate a custom combined loss according to the paper
    https://arxiv.org/abs/1705.07115
    the implementation at the following link was used for reference
    https://github.com/yaringal/multi-task-learning-example/blob/master/ \
    multi-task-learning-example.ipynb
    """

    def __init__(self, config, **kwargs):
        """Initialise the MultiLossLayer.
        Args:
            config (dotmap): Dotmap configuration namespace
        """
        self.config = config
        self.is_placeholder = True
        super(MultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        """Initialise the self.log_vars.
        Args:
            input_shape: Tensor input shape
        """
        self.loss_funcs, self.lw, self.log_vars = [], [], []
        for output in self.config.model.labels:
            if output == data.MAP_NU_TYPE.name:
                self.loss_funcs.append(BinaryCrossentropy(reduction=Reduction.SUM))
                self.log_vars.append(self.add_var(output))
                self.lw.append(1.0)
            elif output == data.MAP_SIGN_TYPE.name:
                self.loss_funcs.append(BinaryCrossentropy(reduction=Reduction.SUM))
                self.log_vars.append(self.add_var(output))
                self.lw.append(1.0)
            elif output == data.MAP_INT_TYPE.name:
                self.loss_funcs.append(
                    SparseCategoricalCrossentropy(reduction=Reduction.SUM)
                )
                self.log_vars.append(self.add_var(output))
                self.lw.append(1.0)
            elif output == data.MAP_ALL_CAT.name:
                self.loss_funcs.append(
                    SparseCategoricalCrossentropy(reduction=Reduction.SUM)
                )
                self.log_vars.append(self.add_var(output))
                self.lw.append(1.0)
            elif output == data.MAP_COSMIC_CAT.name:
                self.loss_funcs.append(BinaryCrossentropy(reduction=Reduction.SUM))
                self.log_vars.append(self.add_var(output))
                self.lw.append(1.0)
            elif output == data.MAP_FULL_COMB_CAT.name:
                self.loss_funcs.append(
                    SparseCategoricalCrossentropy(reduction=Reduction.SUM)
                )
                self.log_vars.append(self.add_var(output))
                self.lw.append(1.0)
            elif output == data.MAP_NU_NC_COMB_CAT.name:
                self.loss_funcs.append(
                    SparseCategoricalCrossentropy(reduction=Reduction.SUM)
                )
                self.log_vars.append(self.add_var(output))
                self.lw.append(1.0)
            elif output == data.MAP_NC_COMB_CAT.name:
                self.loss_funcs.append(
                    SparseCategoricalCrossentropy(reduction=Reduction.SUM)
                )
                self.log_vars.append(self.add_var(output))
                self.lw.append(1.0)
            elif output == "t_nuEnergy":
                self.loss_funcs.append(MeanSquaredError(reduction=Reduction.SUM))
                self.log_vars.append(self.add_var(output))
                self.lw.append(0.0000005)
            elif output in ["t_vtxX", "t_vtxY", "t_vtxZ"]:
                self.loss_funcs.append(MeanSquaredError(reduction=Reduction.SUM))
                self.log_vars.append(self.add_var(output))
                self.lw.append(0.000001)
            if output in ["prim_total", "prim_p", "prim_cp", "prim_np", "prim_g"]:
                self.loss_funcs.append(
                    SparseCategoricalCrossentropy(reduction=Reduction.SUM)
                )
                self.log_vars.append(self.add_var(output))
                self.lw.append(1.0)

        super(MultiLossLayer, self).build(input_shape)

    def add_var(self, output, initial=0.0):
        """Add loss weight variable.
        Args:
            output (str): Output name
            initial (float): Initial weight value
        """
        return self.add_weight(
            name=output + "_log_var",
            shape=(1,),
            dtype=tf.float32,
            initializer=initializers.Constant(initial),
            trainable=True,
        )

    def multi_loss(self, ys_true, ys_pred):
        """Calculate the multi-loss.
        Args:
            ys_true (list[tf.tensor]): True tensors
            ys_pred (list[tf.tensor]): Predicted tensors
        """
        loss = 0
        for i in range(len(ys_true)):
            precision = tf.keras.backend.exp(-self.log_vars[i][0])
            var_loss = self.lw[i] * self.loss_funcs[i](ys_true[i], ys_pred[i])
            loss += tf.keras.backend.sum(precision * var_loss + self.log_vars[i][0])
        return tf.keras.backend.mean(loss)

    def call(self, inputs):
        """Layer call method.
        Args:
            inputs (list[tf.tensor]): List of layer inputs
        """
        num_outputs = len(self.config.model.labels)
        loss = self.multi_loss(inputs[:num_outputs], inputs[num_outputs:])
        self.add_loss(loss, inputs=inputs)
        return tf.constant(1)  # We don't actually use this output
