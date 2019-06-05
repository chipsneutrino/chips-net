import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import regularizers
import utils


def pid_model(categories, input_shape, learning_rate):

    # Structure the sequential model
    model = tf.keras.Sequential([
        layers.Conv2D(filters=64, kernel_size=3, padding='same',
                      activation='relu', input_shape=input_shape),
        layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(0.2),

        layers.Conv2D(filters=128, kernel_size=3,
                      padding='same', activation='relu'),
        layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(0.2),

        layers.Conv2D(filters=256, kernel_size=3,
                      padding='same', activation='relu'),
        layers.Conv2D(filters=256, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(categories, activation='softmax')
    ])

    # Print the model summary
    model.summary()

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=float(learning_rate)),
                  metrics=['accuracy'])

    # Return the compiled model
    return model

# Used for talos optimisation of the pid model


def pid_model_fit(x_train, y_train, x_val, y_val, params):

    # Structure the sequential model
    model = tf.keras.Sequential([
        layers.Conv2D(filters=params["filters_1"], kernel_size=params["size_1"],
                      padding='same', activation='relu', input_shape=(32, 32, 2)),
        layers.Conv2D(
            filters=params["filters_1"], kernel_size=params["size_1"], activation='relu'),
        layers.MaxPooling2D(pool_size=params["pool_1"]),
        layers.Dropout(params["dropout"]),

        layers.Conv2D(
            filters=params["filters_2"], kernel_size=params["size_2"], padding='same', activation='relu'),
        layers.Conv2D(
            filters=params["filters_2"], kernel_size=params["size_2"], activation='relu'),
        layers.MaxPooling2D(pool_size=params["pool_2"]),
        layers.Dropout(params["dropout"]),

        layers.Conv2D(
            filters=params["filters_3"], kernel_size=params["size_3"], padding='same', activation='relu'),
        layers.Conv2D(
            filters=params["filters_3"], kernel_size=params["size_3"], activation='relu'),
        layers.MaxPooling2D(pool_size=params["pool_3"]),
        layers.Dropout(params["dropout"]),

        layers.Flatten(),
        layers.Dense(params["dense"], activation='relu'),
        layers.Dropout(params["dropout"]),
        layers.Dense(params["categories"], activation='softmax')
    ])

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=params["learning_rate"]),
                  metrics=['accuracy'])

    # Fit the model
    history = model.fit(x_train, y_train, batch_size=params["batch_size"],
                        epochs=params["epochs"], verbose=1, validation_data=(x_val, y_val),
                        callbacks=[utils.callback_early_stop("val_acc", params["stop_size"], params["stop_epochs"])])

    # Finally we return the history object and the model
    return history, model


def ppe_model(parameter, input_shape, learning_rate):

    # Structure the sequential model
    model = tf.keras.Sequential([
        layers.Conv2D(filters=32, kernel_size=3, padding='same',
                      activation='relu', input_shape=input_shape),
        layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(filters=64, kernel_size=3,
                      padding='same', activation='relu'),
        layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.0),
        layers.Dense(1, activation='linear')
    ])

    # Print the model summary
    model.summary()

    # Compile the model
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(lr=float(learning_rate)),
                  metrics=['mae', 'mse'])

    # Return the compiled model
    return model

# Used for talos optimisation of parameter models


def ppe_model_fit(x_train, y_train, x_val, y_val, params):

    # Structure the sequential model
    model = tf.keras.Sequential([
        layers.Conv2D(filters=params["filters_1"], kernel_size=params["size_1"],
                      padding='same', activation='relu', input_shape=(32, 32, 2)),
        layers.Conv2D(
            filters=params["filters_1"], kernel_size=params["size_1"], activation='relu'),
        layers.MaxPooling2D(pool_size=params["pool_1"]),
        layers.Conv2D(
            filters=params["filters_2"], kernel_size=params["size_2"], padding='same', activation='relu'),
        layers.Conv2D(
            filters=params["filters_2"], kernel_size=params["size_2"], activation='relu'),
        layers.MaxPooling2D(pool_size=params["pool_2"]),
        layers.Flatten(),
        layers.Dense(params["dense"], activation='relu'),
        layers.Dropout(params["dropout"]),
        layers.Dense(1, activation='linear')
    ])

    # Compile the model
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(lr=params["learning_rate"]),
                  metrics=['mae', 'mse'])

    # Fit the model
    history = model.fit(x_train, y_train, batch_size=params["batch_size"],
                        epochs=params["epochs"], verbose=1, validation_data=(x_val, y_val),
                        callbacks=[utils.callback_early_stop("val_mean_absolute_error", params["stop_size"], params["stop_epochs"])])

    # Finally we return the history object and the model
    return history, model


def par_model(input_shape, learning_rate):

    # Structure the sequential model
    model = tf.keras.Sequential([
        layers.Conv2D(filters=32, kernel_size=3, padding='same',
                      activation='relu', input_shape=input_shape),
        layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(filters=64, kernel_size=3,
                      padding='same', activation='relu'),
        layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.0),
        layers.Dense(8, activation='linear')
    ])

    # Print the model summary
    model.summary()

    # Compile the model
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(lr=float(learning_rate)),
                  metrics=['mae', 'mse'])

    # Return the compiled model
    return model

# Used for talos optimisation of the combined parameter model


def par_model_fit(x_train, y_train, x_val, y_val, params):

    # Structure the sequential model
    model = tf.keras.Sequential([
        layers.Conv2D(filters=params["filters_1"], kernel_size=params["size_1"],
                      padding='same', activation='relu', input_shape=(32, 32, 2)),
        layers.Conv2D(
            filters=params["filters_1"], kernel_size=params["size_1"], activation='relu'),
        layers.MaxPooling2D(pool_size=params["pool_1"]),
        layers.Conv2D(
            filters=params["filters_2"], kernel_size=params["size_2"], padding='same', activation='relu'),
        layers.Conv2D(
            filters=params["filters_2"], kernel_size=params["size_2"], activation='relu'),
        layers.MaxPooling2D(pool_size=params["pool_2"]),
        layers.Flatten(),
        layers.Dense(params["dense"], activation='relu'),
        layers.Dropout(params["dropout"]),
        layers.Dense(8, activation='linear')
    ])

    # Print the model summary
    model.summary()

    # Compile the model
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(lr=params["learning_rate"]),
                  metrics=['mae', 'mse'])

    # Fit the model
    history = model.fit(x_train, y_train, batch_size=params["batch_size"],
                        epochs=params["epochs"], verbose=1, validation_data=(x_val, y_val),
                        callbacks=[utils.callback_early_stop("val_mean_absolute_error", params["stop_size"], params["stop_epochs"])])

    # Finally we return the history object and the model
    return history, model


def Conv2d_All(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = layers.Conv2D(nb_filter, kernel_size, padding=padding,
                      strides=strides, activation='relu', name=conv_name)(x)
    x = layers.BatchNormalization(axis=3, name=bn_name)(x)
    return x


def Inception(x, nb_filter):
    b1x1 = Conv2d_All(x, nb_filter, (1, 1), padding='same',
                      strides=(1, 1), name=None)
    b3x3 = Conv2d_All(x, nb_filter, (1, 1), padding='same',
                      strides=(1, 1), name=None)
    b3x3 = Conv2d_All(b3x3, nb_filter, (3, 3),
                      padding='same', strides=(1, 1), name=None)
    b5x5 = Conv2d_All(x, nb_filter, (1, 1), padding='same',
                      strides=(1, 1), name=None)
    b5x5 = Conv2d_All(b5x5, nb_filter, (1, 1),
                      padding='same', strides=(1, 1), name=None)
    bpool = layers.MaxPool2D(pool_size=(
        3, 3), strides=(1, 1), padding='same')(x)
    bpool = Conv2d_All(bpool, nb_filter, (1, 1),
                       padding='same', strides=(1, 1), name=None)
    x = layers.concatenate([b1x1, b3x3, b5x5, bpool], axis=3)
    return x


def googleNet_model():
    inputs = layers.Input(shape=(32, 32, 2), dtype='float32', name='inputs')
    x = Conv2d_All(inputs, 64, (3, 3), strides=(2, 2), padding='same')
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2d_All(x, 192, (3, 3), strides=(1, 1), padding='same')
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Inception(x, 64)
    x = Inception(x, 120)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Inception(x, 128)
    x = layers.AveragePooling2D(pool_size=(
        2, 2), strides=(2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(0.1))(x)
    #x   = layers.Dense(1024, activation='relu')(x)
    out = layers.Dense(8, activation='linear', name='out')(x)
    model = tf.keras.Model(inputs=inputs, outputs=[out])

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mae', 'mse'])

    return model
