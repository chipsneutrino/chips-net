# Keras Functional API Model Implementations
# Author: Josh Tingey
# Email: j.tingey.16@ucl.ac.uk

import os
from tensorflow import keras
from tensorflow.keras import layers, optimizers, regularizers, callbacks

class BaseModel:
    def __init__(self, config):
        self.config = config

    def summary(self):
        self.model.summary()

    def plot(self):
        file_name = os.path.join("experiments", self.config.exp_name, "summary/plot.png")
        keras.utils.plot_model(self.model, to_file=file_name, show_shapes=True,
                               show_layer_names=True, rankdir='TB',
                               expand_nested=False, dpi=96)

    def compile(self):
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate), 
                           loss=self.loss, metrics=self.metrics)

    def fit(self, train_ds, val_ds):

        callbacks_list = []

        checkpoint_dir = os.path.join("experiments", self.config.exp_name, "checkpoint")
        checkpoint_callback = callbacks.ModelCheckpoint(filepath=checkpoint_dir, verbose=0)
        callbacks_list.append(checkpoint_callback)

        tensorboard_callback = callbacks.TensorBoard(log_dir="tmp", histogram_freq=1)
        callbacks_list.append(tensorboard_callback)

        stopping_callback = callbacks.EarlyStopping(monitor=self.es_monitor, min_delta=self.config.es_delta, 
                                                    patience=self.config.es_epochs, verbose=1, mode='min')
        callbacks_list.append(stopping_callback)
                                            
        self.history = self.model.fit(train_ds, 
                                      epochs=self.config.epochs, 
                                      verbose=1,
                                      validation_data=val_ds,
                                      callbacks = callbacks_list)

    def evaluate(self, test_ds):
        self.model.evaluate(test_ds)


class PIDModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        # Build the model
        inputs = keras.Input(shape=(self.config.img_size, self.config.img_size, 2), name='img')
        x = layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.config.categories, activation='softmax')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='pid_model')

        # Set other parameters
        self.loss = "sparse_categorical_crossentropy"
        self.metrics = ["accuracy"]
        self.es_monitor = "val_acc"

class PPEModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        # Build the model
        inputs = keras.Input(shape=(self.config.img_size, self.config.img_size, 2), name='img')
        x = layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
        x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.0)(x)
        outputs = layers.Dense(1, activation='linear')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='ppe_model')

        # Set other parameters
        self.loss = "mean_squared_error"
        self.metrics = ["mae", "mse"]
        self.es_monitor = "val_mae"

class ParModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        # Build the model
        inputs = keras.Input(shape=(self.config.img_size, self.config.img_size, 2), name='img')
        x = layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
        x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.0)(x)
        outputs = layers.Dense(8, activation='linear')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='par_model')

        # Set other parameters
        self.loss = "mean_squared_error"
        self.metrics = ["mae", "mse"]
        self.es_monitor = "val_mae"