"""Keras Functional API Model Implementations

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module contains the keras functional model definitions. All
are derived from the BaseModel class for ease-of-use in other parts
of the chips-cvn code
"""

import os
from tensorflow import keras
from tensorflow.keras import optimizers, callbacks, Model, Input
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import sherpa


class BaseModel:
    """Base model class which all implementations derive from."""
    def __init__(self, config):
        self.config = config
        self.build()
        self.compile()

    def compile(self):
        """Compiles the model."""
        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.l_rate),
                           loss=self.loss, metrics=self.metrics)

        self.model.summary()  # Print the model summary

        # Plot an image of the model to file
        file_name = os.path.join(self.config.exp_dir, "plot.png")
        keras.utils.plot_model(self.model, to_file=file_name, show_shapes=True,
                               show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

    def fit(self, train_ds, val_ds, extra_callbacks=[]):
        """Given a training and validation dataset, fit the model."""
        callbacks_list = []

        checkpoint = callbacks.ModelCheckpoint(filepath=self.config.exp_dir, save_weights_only=True,
                                               verbose=0)
        callbacks_list.append(checkpoint)

        tensorboard = callbacks.TensorBoard(log_dir="tmp", histogram_freq=1)
        callbacks_list.append(tensorboard)

        stopping = callbacks.EarlyStopping(monitor=self.es_monitor, min_delta=self.config.es_delta,
                                           patience=self.config.es_epochs, verbose=1, mode='min')
        callbacks_list.append(stopping)
        callbacks_list += extra_callbacks

        train_ds = train_ds.batch(self.config.batch_size, drop_remainder=True)
        val_ds = val_ds.batch(self.config.batch_size, drop_remainder=True)

        self.history = self.model.fit(train_ds, epochs=self.config.train_epochs, verbose=1,
                                      validation_data=val_ds, callbacks=callbacks_list)

    def evaluate(self, test_ds):
        """Evaluate the trained model on a test dataset."""
        test_ds = test_ds.batch(self.config.batch_size, drop_remainder=True)
        self.model.evaluate(test_ds)


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
        self.model = Model(inputs=inputs, outputs=outputs, name='ppe_model')

        self.loss = "mean_squared_error"
        self.metrics = ["mae", "mse"]
        self.es_monitor = "val_mae"

    def study_parameters(config):
        """Returns the list of possible parameters for a SHERPA study."""
        pars = [
            sherpa.Ordinal(name='batch_size', range=config.batch_size),
            sherpa.Continuous(name='l_rate', range=config.l_rate, scale='log'),
            sherpa.Ordinal(name='dense_units', range=config.dense_units),
            sherpa.Continuous(name='dropout', range=config.dropout),
            sherpa.Ordinal(name='kernel_size', range=config.kernel_size),
            sherpa.Ordinal(name='filters', range=config.filters),
        ]
        return pars
