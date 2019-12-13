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
        file_name = os.path.join("../experiments", self.config.exp_name, "plot.png")
        keras.utils.plot_model(self.model, to_file=file_name, show_shapes=True,
                               show_layer_names=True, rankdir='TB',
                               expand_nested=False, dpi=96)

    def compile(self):
        self.model.compile(optimizer=optimizers.Adam(lr=self.config.learning_rate), 
                           loss=self.config.loss, metrics=self.config.metrics)

    def fit(self, train_dataset, val_dataset):
        callbacks = [callbacks.ModelCheckpoint(save_weights_only=True, verbose=0),
                     callbacks.TensorBoard(log_dir="tmp", histogram_freq=1),
                     callbacks.EarlyStopping(monitor=self.config.es_monitor, 
                                             min_delta=self.config.es_delta, 
                                             patience=self.config.es_epochs,
                                             verbose=1, mode='min')]
                                            
        self.history = self.model.fit(train_dataset, epochs=self.config.epochs, verbose=1,
                                      validation_data=val_dataset,
                                      callbacks=callbacks)

class PIDModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
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

# 'mean_squared_error' and metrics=['mae', 'mse']
class PPEModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
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

# 'mean_squared_error' and metrics=['mae', 'mse']
class ParModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
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