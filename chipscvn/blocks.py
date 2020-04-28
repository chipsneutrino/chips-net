# -*- coding: utf-8 -*-

"""
Common CNN image classification layers and blocks

This module contains many useful layers and blocks for use in image
classification networks.

An illustrated guide to many of the blocks is at...
https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d
"""

import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers
import tensorflow as tf


def add_inputs(config, core):
    """
    Add reco inputs and apply multi-channel path input logic to a core model block

    Args:
        config (str): Dotmap configuration namespace
        core (func): Core model building function
    Returns:
        tuple (List[tf.keras.Input], tf.tensor): (List of inputs, Keras functional tensor)
    """
    inputs, paths = [], []
    reco_par_names = ['r_vtxX', 'r_vtxY', 'r_vtxZ', 'r_dirTheta', 'r_dirPhi']
    if config.model.reco_pars:
        for name in reco_par_names:
            reco_input = tf.keras.Input(shape=(1), name=name)
            inputs.append(reco_input)
            paths.append(reco_input)

    images = config.data.img_size[2]
    shape = (config.data.img_size[0], config.data.img_size[1], 1)
    if config.data.stack:
        images = 1
        shape = config.data.img_size

    for channel in range(images):
        image = tf.keras.Input(shape=shape, name='image_'+str(channel))
        path = core(image, config, 'path'+str(channel))
        paths.append(path)
        inputs.append(image)

    if len(paths) == 1:
        x = paths[0]
    else:
        x = layers.concatenate(paths, name='reco_pars_concat')
    return inputs, x


def relu6(x):
    """
    Relu6 activation function
    http://www.cs.utoronto.ca/%7Ekriz/conv-cifar10-aug2010.pdf

    Returns:
        tf.keras.activation: Relu6 activation function
    """
    return tf.keras.activations.relu(x, max_value=6.0)


def swish(x):
    """
    Swish activation function
    https://arxiv.org/pdf/1710.05941.pdf

    Returns:
        tf.keras.activation: Swish activation function
    """
    return tf.keras.activations.swish(x)


def conv2d_bn(x, filters, kernel_size=(3, 3), strides=(1, 1), activation='relu',
              padding='same', name=None):
    """
    Utility function to apply conv2d + BatchNormalization.

    Args:
        x (tf.tensor): Input tensor
        filters (int): Number of filters in convolutions
        kernel_size (int): Kernel size in convolutions
        strides (tuple(int, int)): Stride size in convolutions
        activation (str): Activation to use
        padding (str): Padding mode in convolutions
        name (str): Block name prefix
    Returns:
        tf.tensor: Output tensor from `Conv2D` and `BatchNormalization`
    """
    conv_name = None if name is None else name + '_conv'
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      padding=padding, use_bias=False, name=conv_name)(x)
    bn_name = None if name is None else name + '_bn'
    x = layers.BatchNormalization(axis=3, scale=False, name=bn_name)(x)
    ac_name = None if name is None else name + '_ac'
    x = layers.Activation(activation, name=ac_name)(x)
    return x


def depthwise_conv2d_bn(x, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                        padding='same', name=None):
    """
    Utility function to apply depthwise_conv2d + BatchNormalization.

    Args:
        x (tf.tensor): Input tensor
        kernel_size (int): Kernel size in depthwise convolution
        strides (tuple(int, int)): Stride size in depthwise convolution
        activation (str): Activation to use
        padding (str): Padding mode in depthwise convolution
        name (str): Block name prefix
    Returns:
        tf.tensor: Output tensor from `DepthwiseConv2D` and `BatchNormalization`
    """
    dwconv_name = None if name is None else name + '_dwconv'
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, use_bias=False,
                               padding='same', name=dwconv_name)(x)
    bn_name = None if name is None else name + '_bn'
    x = layers.BatchNormalization(axis=3, scale=False, name=bn_name)(x)
    ac_name = None if name is None else name + '_ac'
    x = layers.Activation(activation, name=ac_name)(x)
    return x


def vgg_block(x, num_conv=2, filters=64, kernel_size=(3, 3), strides=(1, 1),
              activation='relu', padding='same', drop_rate=None, bn=True,
              name=None):
    """
    VGG block with added dropout
    https://arxiv.org/pdf/1409.1556.pdf

    Args:
        x (tf.tensor): Input tensor
        num_conv (int): Number of convolutional layers
        filters (int): Number of filters in convolutions
        kernel_size (int): Kernel size in convolutions
        strides (tuple(int, int)): Stride size in convolutions
        activation (str): Activation to use
        padding (str): Padding mode in convolutions
        drop_rate (float): Dropout rate
        bn (bool): Should we use Conv+BN layers
        name (str): Block name prefix
    Returns:
        tf.tensor: Output VGG block tensor
    """
    for i in range(num_conv):
        if bn:
            x = conv2d_bn(x, filters, kernel_size, strides, activation,
                          padding, name=name+'_conv'+str(i))
        else:
            x = layers.Conv2D(filters, kernel_size, activation=activation,
                              padding=padding, name=name+'_conv'+str(i))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=name+'_pool')(x)
    if drop_rate and (drop_rate > 0):  # Dropout if required
        x = layers.Dropout(drop_rate, name=name+'_drop')(x)
    return x


def vgg16_core(x, config, name):
    """
    Core of vgg16 model

    Args:
        x (tf.tensor): Input tensor
        config (dotmap.DotMap): Configuration namespace
        name (str): Block name prefix
    Returns:
        tf.tensor: Output vgg16 core tensor
    """
    x = vgg_block(x, 2, config.model.filters, config.model.kernel_size,
                  drop_rate=config.model.dropout, name=name+'_block1')
    x = vgg_block(x, 2, config.model.filters*2, config.model.kernel_size,
                  drop_rate=config.model.dropout, name=name+'_block2')
    x = vgg_block(x, 3, config.model.filters*4, config.model.kernel_size,
                  drop_rate=config.model.dropout, name=name+'_block3')
    x = vgg_block(x, 3, config.model.filters*8, config.model.kernel_size,
                  drop_rate=config.model.dropout, name=name+'_block4')
    x = vgg_block(x, 3, config.model.filters*8, config.model.kernel_size,
                  drop_rate=config.model.dropout, name=name+'_block5')
    x = layers.Flatten(name='flatten')(x)
    return x


def vgg19_core(x, config, name):
    """
    Core of vgg19 model

    Args:
        x (tf.tensor): Input tensor
        config (dotmap.DotMap): Configuration namespace
        name (str): Block name prefix
    Returns:
        tf.tensor: Output vgg19 core tensor
    """
    x = vgg_block(x, 2, config.model.filters, config.model.kernel_size,
                  drop_rate=config.model.dropout, name=name+'_block1')
    x = vgg_block(x, 2, config.model.filters*2, config.model.kernel_size,
                  drop_rate=config.model.dropout, name=name+'_block2')
    x = vgg_block(x, 4, config.model.filters*4, config.model.kernel_size,
                  drop_rate=config.model.dropout, name=name+'_block3')
    x = vgg_block(x, 4, config.model.filters*8, config.model.kernel_size,
                  drop_rate=config.model.dropout, name=name+'_block4')
    x = vgg_block(x, 4, config.model.filters*8, config.model.kernel_size,
                  drop_rate=config.model.dropout, name=name+'_block5')
    x = layers.Flatten(name='flatten')(x)
    return x


def inception_block(x, filters, strides=(1, 1), activation='relu',
                    padding='same', name=None):
    """
    Returns a Keras functional API Inception Module.
    https://arxiv.org/pdf/1409.4842.pdf

    Args:
        x (tf.tensor): Input tensor
        filters (int): Number of filters in convolutions
        strides (tuple(int, int)): Stride size in convolutions
        activation (str): Activation to use
        padding (str): Padding mode in convolutions
        name (str): Block name prefix
    Returns:
        tf.tensor: Keras functional inceptional module tensor
    """
    b1x1 = layers.Conv2D(filters, (1, 1), activation=activation,
                         padding=padding, name=name+'_1x1')(x)
    b3x3 = layers.Conv2D(filters, (1, 1), activation=activation,
                         padding=padding, name=name+'_3x3_1')(x)
    b3x3 = layers.Conv2D(filters, (3, 3), activation=activation,
                         padding=padding, name=name+'_3x3_2')(b3x3)
    b5x5 = layers.Conv2D(filters, (1, 1), activation=activation,
                         padding=padding, name=name+'_5x5_1')(x)
    b5x5 = layers.Conv2D(filters, (5, 5), activation=activation,
                         padding=padding, name=name+'_5x5_2')(b5x5)
    bpool = layers.MaxPooling2D((3, 3), strides=(1, 1),
                                padding='same', name=name+'_pool')(x)
    bpool = layers.Conv2D(filters, (1, 1), activation=activation,
                          padding=padding, name=name+'_pool_conv')(bpool)
    x = layers.concatenate([b1x1, b3x3, b5x5, bpool], axis=3)
    return x


def inceptionv1_core(x, config, name):
    """
    Core of Inception-v1 model

    Args:
        x (tf.tensor): Input tensor
        config (dotmap.DotMap): Configuration namespace
        name (str): Block name prefix
    Returns:
        tf.tensor: Output Inception-v1 core tensor
    """
    x = conv2d_bn(x, 64, (7, 7), strides=(2, 2), padding='same', name=name+'_conv1')
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name=name+'_pool1')(x)
    x = conv2d_bn(x, 192, (3, 3), strides=(1, 1), padding='same', name=name+'_conv2')
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name=name+'_pool2')(x)
    x = inception_block(x, 64, name=name+'_inception1')
    x = inception_block(x, 120, name=name+'_inception2')
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name=name+'pool3')(x)
    x = inception_block(x, 128, name=name+'_inception3')
    x = inception_block(x, 128, name=name+'_inception4')
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name=name+'_pool4')(x)
    x = inception_block(x, 256, name=name+'_inception5')
    x = layers.AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same',
                                name=name+'_pool5')(x)
    x = layers.Flatten(name=name+'_flatten')(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.1),
                     name=name+'_dense1')(x)
    return x


def mb_conv_block(inputs, kernel_size, in_filters, out_filters, expand_ratio,
                  strides=(1, 1), se_ratio=None, activation='relu', drop_rate=None,
                  name=''):
    """
    Mobile Inverted Residual Bottleneck block with squeeze-and-excitation optimization.

    Args:
        inputs (tf.tensor): Input tensor of conv layer
        kernel_size (int): DepthwiseConv kernel_size
        in_filters (int): Number of input filters
        out_filters (int): Number of output filters
        expand_ratio (int): Filter expansion ratio
        strides (tuple(int, int)): Strides for the convolutions
        se_ratio (float): Squeeze and Excitation ratio
        activation (activation): Which activation to use
        drop_rate (float): Dropout rate
        name (str): Block name prefix
    Returns:
        tf.tensor: Output tensor
    """
    # May want to be using the swish activation, but use relu6 for now, or a leaky relu
    channel_axis = 3
    filters = in_filters * expand_ratio

    # Expansion phase
    x = conv2d_bn(inputs, filters, 1, 1, activation, name=name+'_expand')

    # Depthwise Convolution
    x = depthwise_conv2d_bn(x, kernel_size, strides, activation, name=name)

    # Squeeze and Excitation phase
    if (se_ratio is not None) and (0 < se_ratio <= 1):
        reduced_filters = max(1, int(in_filters * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + '_se_squeeze')(x)
        se = layers.Reshape((1, 1, filters), name=name + '_se_reshape')(se)
        se = layers.Conv2D(reduced_filters, 1, activation=activation, padding='same',
                           use_bias=True, name=name + '_se_reduce')(se)
        se = layers.Conv2D(filters, 1, activation='sigmoid', padding='same',
                           use_bias=True, name=name + '_se_expand')(se)
        x = layers.multiply([x, se], name=name + '_se_excite')

    # Output phase
    x = layers.Conv2D(out_filters, 1, padding='same', use_bias=False, name=name+'_proj_conv')(x)
    x = layers.BatchNormalization(axis=channel_axis, name=name + '_proj_bn')(x)

    if drop_rate and (drop_rate > 0):  # Dropout if required
        x = layers.Dropout(drop_rate, name=name + '_drop')(x)

    if all(s == 1 for s in strides) and in_filters == out_filters:  # Add residual if we can
        x = layers.add([x, inputs], name=name + '_add')

    return x


def effnet_core(x, config, name):
    """
    Core of EfficientNetB0 model

    Args:
        x (tf.tensor): Input tensor
        config (dotmap.DotMap): Configuration namespace
        name (str): Block name name
    Returns:
        tf.tensor: Output Inception-v1 core tensor
    """
    x = mb_conv_block(x, 3, 32, 16, 1, strides=[1, 1], se_ratio=0.25, drop_rate=0.2, name="0")
    x = mb_conv_block(x, 3, 16, 32, 6, strides=[2, 2], se_ratio=0.25, drop_rate=0.2, name="1")
    x = mb_conv_block(x, 3, 32, 64, 6, strides=[2, 2], se_ratio=0.25, drop_rate=0.2, name="2")
    x = mb_conv_block(x, 5, 64, 128, 6, strides=[2, 2], se_ratio=0.25, drop_rate=0.2, name="3")
    x = mb_conv_block(x, 5, 128, 256, 6, strides=[1, 1], se_ratio=0.25, drop_rate=0.2, name="4")
    x = tf.keras.layers.Flatten()(x)
    return x


def get_vgg16_base(config):
    """
    Returns the vgg model base.

    Args:
        config (dotmap.DotMap): Configuration namespace
    Returns:
        tf.tensor: Keras functional tensor
    """
    inputs, x = add_inputs(config, vgg16_core)
    x = layers.Dense(config.model.dense_units, activation='relu', name='dense1')(x)
    x = layers.Dense(config.model.dense_units, activation='relu', name='dense_final')(x)
    x = layers.Dropout(config.model.dropout, name='dropout_final')(x)
    return inputs, x


def get_vgg19_base(config):
    """
    Returns the vgg model base.

    Args:
        config (dotmap.DotMap): Configuration namespace
    Returns:
        tf.tensor: Keras functional tensor
    """
    inputs, x = add_inputs(config, vgg19_core)
    x = layers.Dense(config.model.dense_units, activation='relu', name='dense1')(x)
    x = layers.Dense(config.model.dense_units, activation='relu', name='dense_final')(x)
    x = layers.Dropout(config.model.dropout, name='dropout_final')(x)
    return inputs, x


def get_inceptionv1_base(config):
    """
    Returns the inception model base.

    Args:
        config (dotmap.DotMap): Configuration namespace
    Returns:
        tuple(tf.tensor, List[tf.tensor]): (Keras functional tensor, List of input layers)
    """
    inputs, x = add_inputs(config, inceptionv1_core)
    x = layers.Dense(1024, activation='relu', name='dense2')(x)
    x = layers.Dense(config.model.dense_units, activation='relu', name='dense_final')(x)
    return inputs, x


def get_effnet_base(config):
    """
    Returns the inception model base.

    Args:
        config (dotmap.DotMap): Configuration namespace
    Returns:
        tuple(tf.tensor, List[tf.tensor]): (Keras functional tensor, List of input layers)
    """
    inputs, x = add_inputs(config, effnet_core)
    x = layers.Dense(1024, activation='relu', name='dense2')(x)
    x = layers.Dense(config.model.dense_units, activation='relu', name='dense_final')(x)
    return inputs, x
