# -*- coding: utf-8 -*-

"""Module containing the multitask chipsnet model

An illustrated guide to many of the blocks is at...
https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d
"""

import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.initializers as initializers
import tensorflow as tf


################################
# Kernel Initialisers
################################
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


################################
# Activations
################################
def get_swish():
    def swish(x):
        """Swish activation function: x * sigmoid(x).
        Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
        """
        return tf.keras.activations.swish(x)
    return swish


def get_relu6():
    def relu6(x):
        """relu6 activation function: x * sigmoid(x).
        """
        return tf.keras.activations.relu(x, max_value=6)
    return relu6


################################
# Blocks
################################
class ConvBN(layers.Layer):
    """Convolution + Batch Normalisation layer
    """

    def __init__(self, filters, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu', padding='same', bn=True,
                 prefix='conv_bn', **kwargs):
        """Initialise the ConvBN layer.
        Args:
            filters (int): Number of filters in convolutions
            kernel_size (int): Kernel size in convolutions
            strides (tuple(int, int)): Stride size in convolutions
            activation (str): Activation to use
            padding (str): Padding mode in convolutions
            bn (bool): Shall we apply the batch normalisation?
            prefix (str): Block name prefix
        """
        super(ConvBN, self).__init__(name=prefix, **kwargs)
        self.conv = layers.Conv2D(  # TODO: Study performance of kernel initialiser
            filters,
            kernel_size,
            strides,
            padding,
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=prefix+'_conv')
        self.bn = layers.BatchNormalization(  # TODO: Study performance of scale = True/False
            axis=3,
            scale=True,
            name=prefix+'_bn') if bn else None
        self.activation = layers.Activation(activation, name=prefix+'_ac')

    def call(self, inputs):
        """Run forward pass on ConvBN layer.
        Args:
            inputs (tf.tensor): Input tensor
        Returns:
            tf.tensor: Output tensor from `Conv2D` and `BatchNormalization`
        """
        x = self.conv(inputs)
        x = self.bn(x) if self.bn is not None else x
        x = self.activation(x)
        return x


class DepthwiseConvBN(layers.Layer):
    """Depthwise Convolution + Batch Normalisation layer
    """

    def __init__(self, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu', padding='same', bn=True,
                 prefix='dconv_bn', **kwargs):
        """Initialise the DepthwiseConvBN layer.
        Args:
            kernel_size (int): Kernel size in convolutions
            strides (tuple(int, int)): Stride size in convolutions
            activation (str): Activation to use
            padding (str): Padding mode in convolutions
            bn (bool): Shall we apply the batch normalisation?
            prefix (str): Block name prefix
        """
        super(ConvBN, self).__init__(name=prefix, **kwargs)
        self.dconv = layers.DepthwiseConv2D(  # TODO: Study performance of kernel initialiser
            kernel_size,
            strides,
            padding,
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=prefix+'_dconv')
        self.bn = layers.BatchNormalization(  # TODO: Study performance of scale = True/False
            axis=3,
            scale=True,
            name=prefix+'_bn') if bn else None
        self.activation = layers.Activation(activation, name=prefix+'_ac')

    def call(self, inputs):
        """Run forward pass on DepthwiseConvBN layer.
        Args:
            inputs (tf.tensor): Input tensor
        Returns:
            tf.tensor: Output tensor from `DepthwiseConv2D` and `BatchNormalization`
        """
        x = self.dconv(inputs)
        x = self.bn(x) if self.bn is not None else x
        x = self.activation(x)
        return x


class VGGBlock(layers.Layer):
    """VGG Block layer
    https://arxiv.org/pdf/1409.1556.pdf
    """

    def __init__(self, num_conv=2, filters=64, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu', padding='same', drop_rate=0.0, bn=True,
                 prefix='vgg_block', **kwargs):
        """Initialise the VGGBlock.
        Args:
            num_conv (int): Number of convolutional layers in block
            filters (int): Number of filters in convolutions
            kernel_size (int): Kernel size in convolutions
            strides (tuple(int, int)): Stride size in convolutions
            activation (str): Activation to use
            padding (str): Padding mode in convolutions
            drop_rate (float): Dropout rate to use
            bn (bool): Shall we apply the batch normalisation?
            prefix (str): Block name prefix
        """
        super(VGGBlock, self).__init__(name=prefix, **kwargs)
        self.convs = []
        for i in range(num_conv):
            self.convs.append(ConvBN(
                filters,
                kernel_size,
                strides,
                activation,
                padding,
                bn,
                prefix=prefix+'_conv'+str(i))
            )
        self.pool = layers.MaxPooling2D(
            (2, 2),
            strides=(2, 2),
            name=prefix+'_pool')
        self.dropout = layers.Dropout(
            drop_rate,
            name=prefix+'_drop') if drop_rate > 0.0 else None

    def call(self, inputs):
        """Run forward pass on the VGGBlock.
        Args:
            inputs (tf.tensor): Input tensor
        Returns:
            tf.tensor: Output tensor from VGGBlock
        """
        x = inputs
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        x = self.dropout(x) if self.dropout is not None else x
        return x


class InceptionModule(layers.Layer):
    """Inception Module class
    https://arxiv.org/pdf/1409.4842.pdf
    """

    def __init__(self, filters, strides=(1, 1), activation='relu',
                 padding='same', prefix='conv_bn', **kwargs):
        """Initialise the InceptionModule.
        Args:
            filters (int): Number of filters in convolutions
            strides (tuple(int, int)): Stride size in convolutions
            activation (str): Activation to use
            padding (str): Padding mode in convolutions
            prefix (str): Block name prefix
        """
        super(InceptionModule, self).__init__(name=prefix, **kwargs)
        self.path1_1 = layers.Conv2D(filters, (1, 1), activation=activation,
                                     padding=padding, name=prefix+'_1x1')
        self.path2_1 = layers.Conv2D(filters, (1, 1), activation=activation,
                                     padding=padding, name=prefix+'_3x3_1')
        self.path2_2 = layers.Conv2D(filters, (3, 3), activation=activation,
                                     padding=padding, name=prefix+'_3x3_2')
        self.path3_1 = layers.Conv2D(filters, (1, 1), activation=activation,
                                     padding=padding, name=prefix+'_5x5_1')
        self.path3_2 = layers.Conv2D(filters, (5, 5), activation=activation,
                                     padding=padding, name=prefix+'_5x5_2')
        self.path4_1 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same',
                                           name=prefix+'_pool')
        self.path4_2 = layers.Conv2D(filters, (1, 1), activation=activation,
                                     padding=padding, name=prefix+'_pool_conv')

    def call(self, inputs):
        """Run forward pass on the InceptionModule.
        Args:
            inputs (tf.tensor): Input tensor
        Returns:
            tf.tensor: Output tensor from InceptionModule
        """
        path1 = self.path1_1(inputs)
        path2 = self.path2_1(inputs)
        path2 = self.path2_2(path2)
        path3 = self.path3_1(inputs)
        path3 = self.path3_2(path3)
        path4 = self.path4_1(inputs)
        path4 = self.path4_2(path4)
        x = tf.concat([path1, path2, path3, path4], axis=3)
        return x


class MBConvBlock(layers.Layer):
    """Mobile Inverted Residual Bottleneck block with squeeze-and-excitation optimization.
    """

    def __init__(self, kernel_size, in_filters, out_filters, expand_ratio,
                 strides=(1, 1), se_ratio=None, activation='relu', drop_rate=0.0,
                 prefix='mbconv_block', **kwargs):
        """Initialise the MBConvBlock.
        Args:
            kernel_size (int): DepthwiseConv kernel_size
            in_filters (int): Number of input filters
            out_filters (int): Number of output filters
            expand_ratio (int): Filter expansion ratio
            strides (tuple(int, int)): Strides for the convolutions
            se_ratio (float): Squeeze and Excitation ratio
            activation (activation): Which activation to use
            drop_rate (float): Dropout rate
            prefix (str): Block name prefix
        """
        super(MBConvBlock, self).__init__(name=prefix, **kwargs)
        filters = in_filters * expand_ratio

        # Expansion phase
        self.expansion = ConvBN(
            filters,
            (1, 1),
            (1, 1),
            activation,
            prefix=prefix+'_expand')

        # Depth-wise convolution phase.
        self.depthwise = DepthwiseConvBN(
            kernel_size,
            strides,
            activation,
            prefix=prefix+'_depthwise')

        # Squeeze and Excitation layer.
        self.se_ratio = se_ratio
        if (se_ratio is not None) and (0 < se_ratio <= 1):
            reduced_filters = max(1, int(in_filters * se_ratio))
            self.se_pool = layers.GlobalAveragePooling2D(name=prefix+'_se_squeeze')
            self.se_reshape = layers.Reshape((1, 1, filters), name=prefix+'_se_reshape')
            self.se_reduce = layers.Conv2D(
                reduced_filters,
                kernel_size=(1, 1),
                strides=(1, 1),
                activation=activation,
                padding='same',
                use_bias=True,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=prefix+'_se_reduce')
            self.se_expand = layers.Conv2D(
                filters,
                kernel_size=(1, 1),
                strides=(1, 1),
                activation='sigmoid',
                padding='same',
                use_bias=True,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=prefix+'_se_expand')

        # Output phase.
        self.proj_conv = layers.Conv2D(
            out_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=prefix+'_proj_conv')
        self.proj_bn = layers.BatchNormalization(axis=3, name=prefix+'_proj_bn')
        self.dropout = layers.Dropout(drop_rate, name=prefix+'_drop') if drop_rate > 0.0 else None
        self.residual = all(s == 1 for s in strides) and in_filters == out_filters

    def call(self, inputs):
        """Run forward pass on the MBConvBlock.
        Args:
            inputs (tf.tensor): Input tensor
        Returns:
            tf.tensor: Output tensor from MBConvBlock
        """
        x = self.expansion(inputs)
        x = self.depthwise(x)
        if (self.se_ratio is not None) and (0 < self.se_ratio <= 1):
            se = self.se_pool(x)
            se = self.se_reshape(se)
            se = self.se_reduce(se)
            se = self.se_expand(se)
            x = tf.math.multiply(x, se)
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.dropout(x) if self.dropout is not None else x
        x = tf.add(x, inputs) if self.residual else x  # Add residual if we can
        return x


################################
# Model cores
################################
def vgg16_core(x, config, name):
    """Core of vgg16 model
    Args:
        x (tf.tensor): Input tensor
        config (dotmap.DotMap): Configuration namespace
        name (str): Block name prefix
    Returns:
        tf.tensor: Output vgg16 core tensor
    """
    x = VGGBlock(2, config.model.filters, config.model.kernel_size,
                 drop_rate=config.model.dropout, prefix=name+'_block1')(x)
    x = VGGBlock(2, config.model.filters*2, config.model.kernel_size,
                 drop_rate=config.model.dropout, prefix=name+'_block2')(x)
    x = VGGBlock(3, config.model.filters*4, config.model.kernel_size,
                 drop_rate=config.model.dropout, prefix=name+'_block3')(x)
    x = VGGBlock(3, config.model.filters*8, config.model.kernel_size,
                 drop_rate=config.model.dropout, prefix=name+'_block4')(x)
    x = VGGBlock(3, config.model.filters*8, config.model.kernel_size,
                 drop_rate=config.model.dropout, prefix=name+'_block5')(x)
    x = layers.Flatten(name='flatten')(x)
    return x


def vgg19_core(x, config, name):
    """Core of vgg19 model
    Args:
        x (tf.tensor): Input tensor
        config (dotmap.DotMap): Configuration namespace
        name (str): Block name prefix
    Returns:
        tf.tensor: Output vgg19 core tensor
    """
    x = VGGBlock(2, config.model.filters, config.model.kernel_size,
                 drop_rate=config.model.dropout, prefix=name+'_block1')(x)
    x = VGGBlock(2, config.model.filters*2, config.model.kernel_size,
                 drop_rate=config.model.dropout, prefix=name+'_block2')(x)
    x = VGGBlock(4, config.model.filters*4, config.model.kernel_size,
                 drop_rate=config.model.dropout, prefix=name+'_block3')(x)
    x = VGGBlock(4, config.model.filters*8, config.model.kernel_size,
                 drop_rate=config.model.dropout, prefix=name+'_block4')(x)
    x = VGGBlock(4, config.model.filters*8, config.model.kernel_size,
                 drop_rate=config.model.dropout, prefix=name+'_block5')(x)
    x = layers.Flatten(name='flatten')(x)
    return x


def inceptionv1_core(x, config, name):
    """Core of Inception-v1 model
    Args:
        x (tf.tensor): Input tensor
        config (dotmap.DotMap): Configuration namespace
        name (str): Block name prefix
    Returns:
        tf.tensor: Output Inception-v1 core tensor
    """
    x = ConvBN(64, (7, 7), strides=(2, 2), padding='same', prefix=name+'_conv1')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name=name+'_pool1')(x)
    x = ConvBN(192, (3, 3), strides=(1, 1), padding='same', prefix=name+'_conv2')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name=name+'_pool2')(x)
    x = InceptionModule(64, prefix=name+'_inception1')(x)
    x = InceptionModule(120, prefix=name+'_inception2')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name=name+'pool3')(x)
    x = InceptionModule(128, prefix=name+'_inception3')(x)
    x = InceptionModule(128, prefix=name+'_inception4')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name=name+'_pool4')(x)
    x = InceptionModule(256, prefix=name+'_inception5')(x)
    x = layers.AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same',
                                name=name+'_pool5')(x)
    x = layers.Flatten(name=name+'_flatten')(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.1),
                     name=name+'_dense1')(x)
    return x


def effnet_core(x, config, name):
    """Core of EfficientNetB0 model
    Args:
        x (tf.tensor): Input tensor
        config (dotmap.DotMap): Configuration namespace
        name (str): Block name name
    Returns:
        tf.tensor: Output Inception-v1 core tensor
    """
    x = MBConvBlock(3, 32, 16, 1, strides=[1, 1], se_ratio=0.25, drop_rate=0.2, prefix="0")(x)
    x = MBConvBlock(3, 16, 32, 6, strides=[2, 2], se_ratio=0.25, drop_rate=0.2, prefix="1")(x)
    x = MBConvBlock(3, 32, 64, 6, strides=[2, 2], se_ratio=0.25, drop_rate=0.2, prefix="2")(x)
    x = MBConvBlock(5, 64, 128, 6, strides=[2, 2], se_ratio=0.25, drop_rate=0.2, prefix="3")(x)
    x = MBConvBlock(5, 128, 256, 6, strides=[1, 1], se_ratio=0.25, drop_rate=0.2, prefix="4")(x)
    x = tf.keras.layers.Flatten()(x)
    return x


################################
# Base Models
################################
def get_vgg16_base(config):
    """Returns the vgg model base.
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
    """Returns the vgg model base.
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
    """Returns the inception model base.
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
    """Returns the inception model base.
    Args:
        config (dotmap.DotMap): Configuration namespace
    Returns:
        tuple(tf.tensor, List[tf.tensor]): (Keras functional tensor, List of input layers)
    """
    inputs, x = add_inputs(config, effnet_core)
    x = layers.Dense(1024, activation='relu', name='dense2')(x)
    x = layers.Dense(config.model.dense_units, activation='relu', name='dense_final')(x)
    return inputs, x


################################
# Other
################################
class MultiLossLayer(layers.Layer):
    """Weighted multi-loss layer for multitask network
    https://arxiv.org/pdf/1705.07115.pdf
    https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example.ipynb
    """

    def __init__(self, **kwargs):
        """Initialise the MultiLossLayer.
        Args:
            config (str): Dotmap configuration namespace
        """
        self.is_placeholder = True
        super(MultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        """Initialise the log_vars.
        Args:
            config (str): Dotmap configuration namespace
        """
        self.log_var_c = self.add_weight(
            name='log_var_c', shape=(1,),
            dtype=tf.float32,
            initializer=initializers.Constant(0.),
            # initializer=initializers.RandomUniform(minval=0.2, maxval=1),
            trainable=True
        )
        self.log_var_e = self.add_weight(
            name='log_var_e', shape=(1,),
            dtype=tf.float32,
            initializer=initializers.Constant(0.),
            # initializer=initializers.RandomUniform(minval=0.2, maxval=1),
            trainable=True
        )
        self.loss_func_c = tf.keras.losses.SparseCategoricalCrossentropy()
        # self.loss_func_c = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss_func_e = tf.keras.losses.MeanSquaredError()

        super(MultiLossLayer, self).build(input_shape)

    def multi_loss(self, c_true, e_true, c_pred, e_pred):
        """Calculate the multi-loss.
        Args:
            ys_true (list[tf.tensor]): True tensors
            ys_pred (list[tf.tensor]): Predicted tensors
        """
        # Calculate the categorical loss
        # factor_c = tf.math.divide(1.0, tf.multiply(2.0, self.log_var_c[0]))
        factor_c = tf.math.exp(-self.log_var_c[0])
        loss_c = self.loss_func_c(c_true, c_pred)
        # loss = tf.math.add_n([tf.multiply(factor_c, loss_c), tf.math.log(self.log_var_c[0])])
        loss = tf.math.add_n([tf.multiply(factor_c, loss_c), self.log_var_c[0]])

        # Calculate the energy loss
        # factor_e = tf.math.divide(1.0, tf.multiply(2.0, self.log_var_e[0]))
        factor_e = tf.math.exp(-self.log_var_e[0])
        loss_e = self.loss_func_e(e_true, e_pred)
        # loss = tf.math.add_n([loss, tf.multiply(factor_e, loss_e), tf.math.log(self.log_var_e[0])])
        loss = tf.math.add_n([loss, tf.multiply(factor_e, loss_e), self.log_var_e[0]])

        # Do I need a mean in here?

        return loss

    def call(self, inputs):
        """Layer call method.
        Args:
            inputs (list[tf.tensor]): List of layer inputs
        """
        c_true, e_true = inputs[0], inputs[1]
        c_pred, e_pred = inputs[2], inputs[3]
        loss = self.multi_loss(c_true, e_true, c_pred, e_pred)
        self.add_loss(loss, inputs=inputs)
        return [c_pred, e_pred]


class CHIPSMultitask(tf.keras.Model):
    """Weighted multi-loss layer for multitask network
    https://arxiv.org/pdf/1705.07115.pdf
    https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example.ipynb
    """

    def __init__(self, config, num_cats=16, cat='t_cat', name='chips_multitask', **kwargs):
        super(CHIPSMultitask, self).__init__(name=name, **kwargs)
        self.block1 = VGGBlock(2, config.model.filters, config.model.kernel_size,
                               drop_rate=config.model.dropout, prefix=name+'_block1')
        self.block2 = VGGBlock(2, config.model.filters*2, config.model.kernel_size,
                               drop_rate=config.model.dropout, prefix=name+'_block2')
        self.block3 = VGGBlock(3, config.model.filters*4, config.model.kernel_size,
                               drop_rate=config.model.dropout, prefix=name+'_block3')
        self.block4 = VGGBlock(3, config.model.filters*8, config.model.kernel_size,
                               drop_rate=config.model.dropout, prefix=name+'_block4')
        self.block5 = VGGBlock(3, config.model.filters*8, config.model.kernel_size,
                               drop_rate=config.model.dropout, prefix=name+'_block5')
        self.flatten = layers.Flatten(name='flatten')
        self.dense1 = layers.Dense(config.model.dense_units, activation='relu', name='dense1')
        self.dense2 = layers.Dense(config.model.dense_units, activation='relu', name='dense_final')
        self.dropout = layers.Dropout(config.model.dropout, name='dropout_final')
        self.dense_c = layers.Dense(config.model.dense_units, activation='relu', name='dense_c')
        self.dense_e = layers.Dense(config.model.dense_units, activation='relu', name='dense_e')
        self.out_c = layers.Dense(num_cats, name='logits_c')
        self.out_e = layers.Dense(1, name='logits_e')
        self.out = MultiLossLayer()

    def call(self, inputs):
        x = self.block1(inputs['image_0'])
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x)
        pred_c = self.dense_c(x)
        pred_e = self.dense_e(x)
        pred_c = self.out_c(pred_c)
        pred_e = self.out_e(pred_e)
        return self.out([inputs['t_all_cat'], inputs['t_nuEnergy'], pred_c, pred_e])

    def model(self):
        image = layers.Input(shape=(64, 64, 3), name='image_0')
        true_c = tf.keras.Input(shape=(1), name='true_c')
        true_e = layers.Input(shape=(1), name='true_e')
        return tf.keras.Model(
            inputs=[image, true_c, true_e],
            outputs=self.call({
                'image_0': image,
                't_all_cat': true_c,
                't_nuEnergy': true_e
            })
        )


def add_inputs(config, core):
    """Add reco inputs and apply multi-channel path input logic to a core model block
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
