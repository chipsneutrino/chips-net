"""Keras Functional API Model Implementations

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module contains the keras functional model definitions. All
are derived from the BaseModel class for ease-of-use in other parts
of the chips-cvn code
"""

import os
from tensorflow.keras import optimizers, Model, Input
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, concatenate
import tensorflow as tf


class BaseModel:
    """Base model class which all model implementations derive from."""
    def __init__(self, config):
        self.config = config
        self.build()

    def load(self, checkpoint_num=None):
        """Returns the correct model with its trained weights loaded."""
        if checkpoint_num is None:
            latest = tf.train.latest_checkpoint(self.config.exp.checkpoints_dir)
            self.model.load_weights(latest).expect_partial()
        else:
            checkpoint_path = self.config.exp.checkpoints_dir + 'cp-' + str(checkpoint_num).zfill(4) + '.ckpt'
            self.model.load_weights(checkpoint_path).expect_partial()

    def summarise(self):
        """Prints model to stdout and plots diagram of model to file."""
        self.model.summary()  # Print the model structure to stdout

        # Plot an image of the model to file
        file_name = os.path.join(self.config.exp.exp_dir, 'model_diagram.png')
        tf.keras.utils.plot_model(self.model, to_file=file_name, show_shapes=True,
                                  show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

    def build(self):
        """Build the model, overide in derived model class."""
        raise NotImplementedError


class SingleParModel(BaseModel):
    """Single parameter estimation model class."""
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API."""
        
        ct_input = Input(shape=(64,64,2), name='ct_image')
        x = Conv2D(self.config.model.filters, self.config.model.kernel_size, padding='same', activation='relu')(ct_input)
        x = Conv2D(self.config.model.filters, self.config.model.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.model.dropout)(x)
        x = Conv2D((self.config.model.filters*2), self.config.model.kernel_size, padding='same', activation='relu')(x)
        x = Conv2D((self.config.model.filters*2), self.config.model.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.model.dropout)(x)
        x = Conv2D((self.config.model.filters*4), self.config.model.kernel_size, padding='same', activation='relu')(x)
        x = Conv2D((self.config.model.filters*4), self.config.model.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.model.dropout)(x)
        x = Conv2D((self.config.model.filters*8), self.config.model.kernel_size, padding='same', activation='relu')(x)
        x = Conv2D((self.config.model.filters*8), self.config.model.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.model.dropout)(x)
        x = Flatten()(x)

        h_input = Input(shape=(64,64,1), name='h_image')
        y = Conv2D(self.config.model.filters, self.config.model.kernel_size, padding='same', activation='relu')(h_input)
        y = Conv2D(self.config.model.filters, self.config.model.kernel_size, activation='relu')(y)
        y = MaxPooling2D(pool_size=2)(y)
        y = Dropout(self.config.model.dropout)(y)
        y = Conv2D((self.config.model.filters*2), self.config.model.kernel_size, padding='same', activation='relu')(y)
        y = Conv2D((self.config.model.filters*2), self.config.model.kernel_size, activation='relu')(y)
        y = MaxPooling2D(pool_size=2)(y)
        y = Dropout(self.config.model.dropout)(y)
        y = Conv2D((self.config.model.filters*4), self.config.model.kernel_size, padding='same', activation='relu')(y)
        y = Conv2D((self.config.model.filters*4), self.config.model.kernel_size, activation='relu')(y)
        y = MaxPooling2D(pool_size=2)(y)
        y = Dropout(self.config.model.dropout)(y)
        y = Conv2D((self.config.model.filters*8), self.config.model.kernel_size, padding='same', activation='relu')(y)
        y = Conv2D((self.config.model.filters*8), self.config.model.kernel_size, activation='relu')(y)
        y = MaxPooling2D(pool_size=2)(y)
        y = Dropout(self.config.model.dropout)(y)
        y = Flatten()(y)
        
        vtxX_input = Input(shape=(1), name='reco_vtxX')
        vtxY_input = Input(shape=(1), name='reco_vtxY')
        vtxZ_input = Input(shape=(1), name='reco_vtxZ')
        dirTheta_input = Input(shape=(1), name='reco_dirTheta')
        dirPhi_input = Input(shape=(1), name='reco_dirPhi')
        x = concatenate([x, y, vtxX_input, vtxY_input, vtxZ_input, dirTheta_input, dirPhi_input])

        x = Dense(self.config.model.dense_units, activation='relu')(x)
        x = Dense(self.config.model.dense_units, activation='relu')(x)
        x = Dense(self.config.model.dense_units, activation='relu', name='dense_final')(x)
        x = Dropout(self.config.model.dropout)(x)
        outputs = Dense(1, activation='linear', name=self.config.model.parameter)(x)
        self.model = Model(inputs=[ct_input, h_input, vtxX_input, vtxY_input, vtxZ_input, dirTheta_input, dirPhi_input], 
                           outputs=outputs, name='single_par_model')

        self.loss = 'mean_squared_error'
        self.loss_weights = None
        self.metrics = ['mae', 'mse']
        self.es_monitor = 'val_mae'
        self.parameters = [self.config.model.parameter]

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           loss_weights=self.loss_weights,
                           metrics=self.metrics)


class ClassificationModel(BaseModel):
    """Event classification model class."""
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API."""
        ct_input = Input(shape=(64,64,2), name='ct_image')
        x = Conv2D(self.config.model.filters, self.config.model.kernel_size, padding='same', activation='relu')(ct_input)
        x = Conv2D(self.config.model.filters, self.config.model.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.model.dropout)(x)
        x = Conv2D((self.config.model.filters*2), self.config.model.kernel_size, padding='same', activation='relu')(x)
        x = Conv2D((self.config.model.filters*2), self.config.model.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.model.dropout)(x)
        x = Conv2D((self.config.model.filters*4), self.config.model.kernel_size, padding='same', activation='relu')(x)
        x = Conv2D((self.config.model.filters*4), self.config.model.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.model.dropout)(x)
        x = Conv2D((self.config.model.filters*8), self.config.model.kernel_size, padding='same', activation='relu')(x)
        x = Conv2D((self.config.model.filters*8), self.config.model.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.model.dropout)(x)
        x = Flatten()(x)

        h_input = Input(shape=(64,64,1), name='h_image')
        y = Conv2D(self.config.model.filters, self.config.model.kernel_size, padding='same', activation='relu')(h_input)
        y = Conv2D(self.config.model.filters, self.config.model.kernel_size, activation='relu')(y)
        y = MaxPooling2D(pool_size=2)(y)
        y = Dropout(self.config.model.dropout)(y)
        y = Conv2D((self.config.model.filters*2), self.config.model.kernel_size, padding='same', activation='relu')(y)
        y = Conv2D((self.config.model.filters*2), self.config.model.kernel_size, activation='relu')(y)
        y = MaxPooling2D(pool_size=2)(y)
        y = Dropout(self.config.model.dropout)(y)
        y = Conv2D((self.config.model.filters*4), self.config.model.kernel_size, padding='same', activation='relu')(y)
        y = Conv2D((self.config.model.filters*4), self.config.model.kernel_size, activation='relu')(y)
        y = MaxPooling2D(pool_size=2)(y)
        y = Dropout(self.config.model.dropout)(y)
        y = Conv2D((self.config.model.filters*8), self.config.model.kernel_size, padding='same', activation='relu')(y)
        y = Conv2D((self.config.model.filters*8), self.config.model.kernel_size, activation='relu')(y)
        y = MaxPooling2D(pool_size=2)(y)
        y = Dropout(self.config.model.dropout)(y)
        y = Flatten()(y)
        
        vtxX_input = Input(shape=(1), name='reco_vtxX')
        vtxY_input = Input(shape=(1), name='reco_vtxY')
        vtxZ_input = Input(shape=(1), name='reco_vtxZ')
        dirTheta_input = Input(shape=(1), name='reco_dirTheta')
        dirPhi_input = Input(shape=(1), name='reco_dirPhi')
        x = concatenate([x, y, vtxX_input, vtxY_input, vtxZ_input, dirTheta_input, dirPhi_input])

        x = Dense(self.config.model.dense_units, activation='relu')(x)
        x = Dense(self.config.model.dense_units, activation='relu')(x)
        x = Dense(self.config.model.dense_units, activation='relu', name='dense_final')(x)
        x = Dropout(self.config.model.dropout)(x)
        outputs = Dense(self.config.model.categories, activation='softmax', name='true_category')(x)
        self.model = Model(inputs=[ct_input, h_input, vtxX_input, vtxY_input, vtxZ_input, dirTheta_input, dirPhi_input], 
                           outputs=outputs, name='classification_model')

        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.es_monitor = 'val_accuracy'
        self.parameters = ['true_category']

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           metrics=self.metrics)


class MultiTaskModel(BaseModel):
    """Multi-task model class."""
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """Builds the model using the keras functional API."""
        ct_input = Input(shape=(64,64,2), name='ct_image')
        x = Conv2D(self.config.model.filters, self.config.model.kernel_size, padding='same', activation='relu')(ct_input)
        x = Conv2D(self.config.model.filters, self.config.model.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.model.dropout)(x)
        x = Conv2D((self.config.model.filters*2), self.config.model.kernel_size, padding='same', activation='relu')(x)
        x = Conv2D((self.config.model.filters*2), self.config.model.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.model.dropout)(x)
        x = Conv2D((self.config.model.filters*4), self.config.model.kernel_size, padding='same', activation='relu')(x)
        x = Conv2D((self.config.model.filters*4), self.config.model.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.model.dropout)(x)
        x = Conv2D((self.config.model.filters*8), self.config.model.kernel_size, padding='same', activation='relu')(x)
        x = Conv2D((self.config.model.filters*8), self.config.model.kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(self.config.model.dropout)(x)
        x = Flatten()(x)

        h_input = Input(shape=(64,64,1), name='h_image')
        y = Conv2D(self.config.model.filters, self.config.model.kernel_size, padding='same', activation='relu')(h_input)
        y = Conv2D(self.config.model.filters, self.config.model.kernel_size, activation='relu')(y)
        y = MaxPooling2D(pool_size=2)(y)
        y = Dropout(self.config.model.dropout)(y)
        y = Conv2D((self.config.model.filters*2), self.config.model.kernel_size, padding='same', activation='relu')(y)
        y = Conv2D((self.config.model.filters*2), self.config.model.kernel_size, activation='relu')(y)
        y = MaxPooling2D(pool_size=2)(y)
        y = Dropout(self.config.model.dropout)(y)
        y = Conv2D((self.config.model.filters*4), self.config.model.kernel_size, padding='same', activation='relu')(y)
        y = Conv2D((self.config.model.filters*4), self.config.model.kernel_size, activation='relu')(y)
        y = MaxPooling2D(pool_size=2)(y)
        y = Dropout(self.config.model.dropout)(y)
        y = Conv2D((self.config.model.filters*8), self.config.model.kernel_size, padding='same', activation='relu')(y)
        y = Conv2D((self.config.model.filters*8), self.config.model.kernel_size, activation='relu')(y)
        y = MaxPooling2D(pool_size=2)(y)
        y = Dropout(self.config.model.dropout)(y)
        y = Flatten()(y)
        
        vtxX_input = Input(shape=(1), name='reco_vtxX')
        vtxY_input = Input(shape=(1), name='reco_vtxY')
        vtxZ_input = Input(shape=(1), name='reco_vtxZ')
        dirTheta_input = Input(shape=(1), name='reco_dirTheta')
        dirPhi_input = Input(shape=(1), name='reco_dirPhi')
        x = concatenate([x, y, vtxX_input, vtxY_input, vtxZ_input, dirTheta_input, dirPhi_input])

        x = Dense(self.config.model.dense_units, activation='relu')(x)
        x = Dense(self.config.model.dense_units, activation='relu')(x)
        x = Dense(self.config.model.dense_units, activation='relu', name='dense_final')(x)
        x = Dropout(self.config.model.dropout)(x)
        pdg_path = Dense(self.config.model.dense_units, activation='relu')(x)
        type_path = Dense(self.config.model.dense_units, activation='relu')(x)
        energy_path = Dense(self.config.model.dense_units, activation='relu')(x)
        pdg_output = Dense(self.config.model.pdgs, activation='softmax', name='true_pdg')(pdg_path)
        type_output = Dense(self.config.model.types, activation='softmax', name='true_type')(type_path)
        energy_output = Dense(1, activation='linear', name='true_nuEnergy')(energy_path)
        self.model = Model(inputs=[ct_input, h_input, vtxX_input, vtxY_input, vtxZ_input, dirTheta_input, dirPhi_input],
                           outputs=[pdg_output, type_output, energy_output], name='multi_task_model')

        self.loss = {
            'pdg': 'sparse_categorical_crossentropy',
            'type': 'sparse_categorical_crossentropy',
            'nuEnergy': 'mean_squared_error',
        }
        self.loss_weights = {
            'pdg': 1.0,
            'type': 1.0,
            'nuEnergy': 1.0
        }
        self.metrics = {
            'pdg': 'accuracy',
            'type': 'accuracy',
            'nuEnergy': 'mae'
        }
        self.es_monitor = 'val_accuracy'
        self.parameters = ['pdg', 'type', 'nuEnergy']

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.model.lr),
                           loss=self.loss,
                           loss_weights=self.loss_weights,
                           metrics=self.metrics)
