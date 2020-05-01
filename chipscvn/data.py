# -*- coding: utf-8 -*-

"""Data creation and loading module

This module contains both the DataCreator and DataLoader classes, these
are used to firstly generate tfrecords files from ROOT hitmap files and
then to read these on the fly using tf.datasets at model training or
evaluation.
"""

import os
from joblib import Parallel, delayed
import multiprocessing
import random

import pandas as pd
import uproot
import numpy as np
import tensorflow as tf
from dotmap import DotMap


"""Map nuel and numu (Total = 2)
0=Nuel, 1=Numu (cosmic muons are included in this)"""
nu_type_map = DotMap({
    'name': 't_nu_type',
    'total_num': 2,
    'train_num': 2,
    'labels': ['Nuel', 'Numu'],
    'table': tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant([11, 12, 13, 14]),
            tf.constant([0,  0,  1,  1])
        ), -1)
})

"""Map interaction types (Total = 10)
0=CC-QEL, 1=CC-RES, 2=CC-DIS, 3=CC-COH
4=NC-QEL, 5=NC-RES, 6=NC-DIS, 7=NC-COH, 8=Cosmic, 9=Other"""
int_type_map = DotMap({
    'name': 't_int_type',
    'total_num': 10,
    'train_num': 8,
    'labels': ['CC-QEL', 'CC-RES', 'CC-DIS', 'CC-COH',
               'NC-QEL', 'NC-RES', 'NC-DIS', 'NC-COH', 'Cosmic', 'Other'],
    'table': tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 91, 92, 96, 97, 98, 99, 100]),
            tf.constant([9, 0, 4, 1, 1, 1, 5, 5, 5, 5,  2,  6,  7,  3,  9,  9,   8])
        ), -1)
})

"""Map to all categories (Total = 19)
Category keys are a string of pdg+type, e.g an nuel ccqe event is '0'+'0' = '00'
0=Nuel-CC-QEL, 1=Nuel-CC-RES, 2=Nuel-CC-DIS, 3=Nuel-CC-COH
4=Numu-CC-QEL, 5=Numu-CC-RES, 6=Numu-CC-DIS, 7=Numu-CC-COH
8=Nuel-NC-QEL, 9=Nuel-NC-RES, 10=Nuel-NC-DIS, 11=Nuel-NC-COH
12=Numu-NC-QEL, 13=Numu-NC-RES, 14=Numu-NC-DIS, 15=Numu-NC-COH
16=Cosmic, 17=Nuel-Other, 18=Numu-Other"""
all_cat_map = DotMap({
    'name': 't_all_cat',
    'total_num': 19,
    'train_num': 16,
    'labels': ['Nuel-CC-QEL', 'Nuel-CC-RES', 'Nuel-CC-DIS', 'Nuel-CC-COH'
               'Numu-CC-QEL', 'Numu-CC-RES', 'Numu-CC-DIS', 'Numu-CC-COH'
               'Nuel-NC-QEL', 'Nuel-NC-RES', 'Nuel-NC-DIS', 'Nuel-NC-COH'
               'Numu-NC-QEL', 'Numu-NC-RES', 'Numu-NC-DIS', 'Numu-NC-COH'
               'Cosmic', 'Nuel-Other', 'Numu-Other'],
    'table': tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(['00', '01', '02', '03', '10', '11', '12', '13',
                         '04', '05', '06', '07', '14', '15', '16', '17',
                         '18', '09', '19'], dtype=tf.string),
            tf.constant([0, 1, 2, 3, 4, 5, 6, 7,
                         8, 9, 10, 11, 12, 13, 14, 15,
                         16, 17, 18])
        ), -1)
})

"""Map a cosmic flag (Total = 2)
0=Cosmic, 1=Not-Cosmic"""
cosmic_cat_map = DotMap({
    'name': 't_cosmic_cat',
    'total_num': 2,
    'train_num': 2,
    'labels': ['Cosmic', 'Not-Cosmic'],
    'table': tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant([0, 1, 2, 3, 4, 5, 6, 7,
                         8, 9, 10, 11, 12, 13, 14, 15,
                         16, 17, 18]),
            tf.constant([0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         1, 0, 0])
        ), -1)
})

"""Map to full_combined categories (Total = 5)
0=Nuel-CC, 1=Numu-CC, 2=NC, 3=Cosmic, 4=Other"""
comb_cat_map = DotMap({
    'name': 't_comb_cat',
    'total_num': 5,
    'train_num': 3,
    'labels': ['Nuel-CC', 'Numu-CC', 'NC', 'Cosmic', 'Other'],
    'table': tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant([0, 1, 2, 3, 4, 5, 6, 7,
                         8, 9, 10, 11, 12, 13, 14, 15,
                         16, 17, 18]),
            tf.constant([0, 0, 0, 0, 1, 1, 1, 1,
                         2, 2, 2, 2, 2, 2, 2, 2,
                         3, 4, 4])
        ), -1)
})

"""Map to nc_nu_combined categories (Total = 14)
0=Nuel CC-QEL, 1=Nuel CC-RES, 2=Nuel CC-DIS, 3=Nuel CC-COH
4=Numu CC-QEL, 5=Numu CC-RES, 6=Numu CC-DIS, 7=Numu CC-COH
8=NC-QEL, 9=NC-RES, 10=NC-DIS, 11=NC-COH, 12=Cosmic, 13=Other"""
nu_nc_comb_map = DotMap({
    'name': 't_nu_nc_cat',
    'total_num': 14,
    'train_num': 12,
    'labels': ['Nuel CC-QEL', 'Nuel CC-RES', 'Nuel CC-DIS', 'Nuel CC-COH',
               'Numu CC-QEL', 'Numu CC-RES', 'Numu CC-DIS', 'Numu CC-COH',
               'NC-QEL', 'NC-RES', 'NC-DIS', 'NC-COH', 'Cosmic', 'Other'],
    'table': tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant([0, 1, 2, 3, 4, 5, 6, 7,
                         8, 9, 10, 11, 12, 13, 14, 15,
                         16, 17, 18]),
            tf.constant([0, 1, 2, 3, 4, 5, 6, 7,
                         8, 9, 10, 11, 8, 9, 10, 11,
                         12, 13, 13])
        ), -1)
})

"""Map to nc_combined categories (Total = 11)
0=Nuel-CC-QEL, 1=Nuel-CC-RES, 2=Nuel-CC-DIS, 3=Nuel-CC-COH
4=Numu-CC-QEL, 5=Numu-CC-RES, 6=Numu-CC-DIS, 7=Numu-CC-COH
8=NC, 9=Cosmic, 10=Other"""
nc_comb_map = DotMap({
    'name': 't_nc_cat',
    'total_num': 11,
    'train_num': 9,
    'labels': ['Nuel-CC-QEL', 'Nuel-CC-RES', 'Nuel-CC-DIS', 'Nuel-CC-COH',
               'Numu-CC-QEL', 'Numu-CC-RES', 'Numu-CC-DIS', 'Numu-CC-COH',
               'NC', 'Cosmic', 'Other'],
    'table': tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant([0, 1, 2, 3, 4, 5, 6, 7,
                         8, 9, 10, 11, 12, 13, 14, 15,
                         16, 17, 18]),
            tf.constant([0, 1, 2, 3, 4, 5, 6, 7,
                         8, 8, 8, 8, 8, 8, 8, 8,
                         9, 10, 10])
        ), -1)
})


def get_map(name):
    """Getting category mapping dict from name.
    Args:
        name (str): Name of mapping
    Returns:
        dict: Mapping dictionary
    """
    for map in [nu_type_map, int_type_map, all_cat_map, cosmic_cat_map,
                comb_cat_map, nu_nc_comb_map, nc_comb_map]:
        if map.name == name:
            return map
    return None


class DataLoader:
    """Generates tf datasets for training/evaluation from the configuration.
    """

    def __init__(self, config):
        """Initialise the DataLoader.
        Args:
            config (str): Dotmap configuration namespace
        """
        self.config = config
        self.train_dirs = [os.path.join(in_dir, 'train') for in_dir in config.data.input_dirs]
        self.val_dirs = [os.path.join(in_dir, 'val') for in_dir in config.data.input_dirs]
        self.test_dirs = [os.path.join(in_dir, 'test') for in_dir in config.data.input_dirs]

        if config.data.all_chan:
            self.full_image_shape = [64, 64, 13]
        else:
            self.full_image_shape = [64, 64, 3]

        self.rand = []
        self.shift = []
        for i, enabled in enumerate(self.config.data.channels):
            self.rand.append(tf.random.normal(
                shape=[64, 64], mean=1, stddev=config.data.rand[i], dtype=tf.float32))
            self.shift.append(tf.fill([64, 64], (1.0 + config.data.shift[i])))

    @tf.function
    def parse(self, serialised_example):
        """Parses a single serialised example into both an input and labels dict.
        Args:
            serialised_example (tf.Example): A single example from .tfrecords file
        Returns:
            Tuple[dict, dict]: (Inputs dictionary, Labels dictionary)
        """
        features = {
            'true_pars_i': tf.io.FixedLenFeature([], tf.string),
            'true_pars_f': tf.io.FixedLenFeature([], tf.string),
            'true_prim_i': tf.io.FixedLenFeature([], tf.string),
            'true_prim_f': tf.io.FixedLenFeature([], tf.string),
            'reco_pars_i': tf.io.FixedLenFeature([], tf.string),
            'reco_pars_f': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(serialised_example, features)

        inputs, labels = {}, {}  # The two dictionaries to fill

        # We first generate the core inputs and labels we need for training.
        # Extra variables are then added later if required
        # Decode and reshape the 'image' into a tf tensor and then 'unstack'
        full_image = tf.io.decode_raw(example['image'], tf.uint8)
        full_image = tf.reshape(full_image, self.full_image_shape)
        unstacked = tf.unstack(full_image, axis=2)

        channels = []
        for i, enabled in enumerate(self.config.data.channels):
            if enabled:
                # Cast to float, scale to [0,1], apply rand, apply shift
                unstacked[i] = tf.cast(unstacked[i], tf.float32) / 256.0
                unstacked[i] = tf.math.multiply(unstacked[i], self.rand[i])
                unstacked[i] = tf.math.multiply(unstacked[i], self.shift[i])
                channels.append(unstacked[i])
                # TODO: Could take values below zero, change to prevent this

        # Choose to either stack the channels back into a single tensor or keep them seperate
        if self.config.data.stack:
            inputs['image_0'] = tf.stack(channels, axis=2)
        else:
            for i, input_image in enumerate(channels):
                inputs['image_'+str(i)] = tf.expand_dims(input_image, 2)

        # Generate all the category mappings
        true_pars_i = tf.io.decode_raw(example['true_pars_i'], tf.int32)
        nu_type = nu_type_map.table.lookup(true_pars_i[0])
        labels[nu_type_map.name] = nu_type
        int_type = int_type_map.table.lookup(true_pars_i[1])
        labels[int_type_map.name] = int_type
        category = all_cat_map.table.lookup(
            tf.strings.join((
                tf.strings.as_string(nu_type),
                tf.strings.as_string(int_type)
            ))
        )
        labels[all_cat_map.name] = category
        labels[cosmic_cat_map.name] = cosmic_cat_map.table.lookup(category)
        labels[comb_cat_map.name] = comb_cat_map.table.lookup(category)
        labels[nu_nc_comb_map.name] = nu_nc_comb_map.table.lookup(category)
        labels[nc_comb_map.name] = nc_comb_map.table.lookup(category)

        true_pars_f = tf.io.decode_raw(example['true_pars_f'], tf.float32)
        labels['t_vtxX'] = true_pars_f[0]
        labels['t_vtxY'] = true_pars_f[1]
        labels['t_vtxZ'] = true_pars_f[2]
        labels['t_vtxT'] = true_pars_f[3]
        labels['t_nuEnergy'] = true_pars_f[4]

        reco_pars_f = tf.io.decode_raw(example['reco_pars_f'], tf.float32)
        inputs['r_vtxX'] = tf.math.divide(reco_pars_f[4], self.config.data.par_scale[0]),
        inputs['r_vtxY'] = tf.math.divide(reco_pars_f[5], self.config.data.par_scale[1]),
        inputs['r_vtxZ'] = tf.math.divide(reco_pars_f[6], self.config.data.par_scale[2]),
        inputs['r_dirTheta'] = tf.math.divide(reco_pars_f[8], self.config.data.par_scale[3]),
        inputs['r_dirPhi'] = tf.math.divide(reco_pars_f[9], self.config.data.par_scale[4])

        if len(self.config.model.labels) > 1:
            inputs[all_cat_map.name] = labels[all_cat_map.name]
            inputs['t_nuEnergy'] = labels['t_nuEnergy']

        if self.config.data.extra_vars:
            true_prim_i = tf.io.decode_raw(example['true_prim_i'], tf.int32)
            true_prim_f = tf.io.decode_raw(example['true_prim_f'], tf.float32)
            reco_pars_i = tf.io.decode_raw(example['reco_pars_i'], tf.int32)

            # Need to reshape the primary particle array
            true_prim_f = tf.reshape(true_prim_f, [3, 10])

            labels['t_p_pdgs'] = true_prim_i
            labels['t_p_energies'] = true_prim_f[0]
            labels['t_p_dirTheta'] = true_prim_f[1]
            labels['t_p_dirPhi'] = true_prim_f[2]

            inputs['r_raw_num_hits'] = reco_pars_i[0]
            inputs['r_filtered_num_hits'] = reco_pars_i[1]
            inputs['r_num_hough_rings'] = reco_pars_i[2]
            inputs['r_raw_total_digi_q'] = reco_pars_f[0]
            inputs['r_filtered_total_digi_q'] = reco_pars_f[1]
            inputs['r_first_ring_height'] = reco_pars_f[2]
            inputs['r_last_ring_height'] = reco_pars_f[3]
            inputs['r_vtxT'] = reco_pars_f[7]

        return inputs, labels

    def filter_other(self, inputs, labels):
        """Filters out 'other' cateogory events from dataset.
        Args:
            inputs (dict): Inputs dictionary
            labels (dict): Labels dictionary
        Returns:
            bool: Is this an 'other' category event?
        """
        if (labels[all_cat_map.name]) == 17 or (labels[all_cat_map.name] == 18):
            return False
        else:
            return True

    def dataset(self, dirs, parallel=True):
        """Returns a dataset formed from all the files in the input directories.
        Args:
            dirs (list[str]): List of input directories
        Returns:
            tf.dataset: The generated dataset
        """
        files = []  # Add all files in dirs to a list
        for d in dirs:
            for file in os.listdir(d):
                files.append(os.path.join(d, file))

        random.seed(8)
        random.shuffle(files)  # Shuffle the list to randomise the 'interleave'
        ds = tf.data.Dataset.from_tensor_slices(files)

        if parallel:
            ds = ds.interleave(
                tf.data.TFRecordDataset,
                cycle_length=len(files),
                block_length=1,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            ds = ds.map(lambda x: self.parse(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            ds = ds.interleave(
                tf.data.TFRecordDataset,
                cycle_length=1,
                block_length=1
            )
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            ds = ds.map(lambda x: self.parse(x))

        ds = ds.filter(self.filter_other)
        ds = ds.batch(self.config.data.batch_size, drop_remainder=True)

        return ds

    def df_from_ds(self, df):
        """Create a pandas dataframe from a tf dataset
        Args:
            df (tf.dataset): Input dataset
        Returns:
            pandas.DataFrame: DataFrame generated from the dataset
        """
        events = {}
        for x, y in df:
            for name, array in list(x.items()):  # Fill events dict with 'inputs'
                if name in events.keys():
                    events[name].extend(array.numpy())
                else:
                    events[name] = []
                    events[name].extend(array.numpy())

            for name, array in list(y.items()):  # Fill events dict with 'labels'
                if name in events.keys():
                    events[name].extend(array.numpy())
                else:
                    events[name] = []
                    events[name].extend(array.numpy())

        return pd.DataFrame.from_dict(events)  # Convert dict to pandas dataframe

    @property
    def training_ds(self):
        """Returns the training dataset.
        Returns:
            tf.dataset: The training dataset
        """
        ds = self.dataset(self.train_dirs)
        ds = ds.take(self.config.data.train_examples)
        return ds

    @property
    def validation_ds(self):
        """Returns the validation dataset.
        Returns:
            tf.dataset: The validation dataset
        """
        ds = self.dataset(self.val_dirs)
        ds = ds.take(int(self.config.data.val_examples))
        return ds

    @property
    def testing_ds(self, parallel=False):
        """Returns the testing dataset.
        Returns:
            tf.dataset: The testing dataset
        """
        ds = self.dataset(self.test_dirs)
        ds = ds.take(self.config.data.test_examples)
        return ds

    @property
    def training_df(self):
        """Returns the training DataFrame.
        Returns:
            pd.DataFrame: Training data DataFrame
        """
        return self.df_from_ds(self.training_ds)

    @property
    def validation_df(self):
        """Returns the validation DataFrame.
        Returns:
            pd.DataFrame: Validation data DataFrame
        """
        return self.df_from_ds(self.validation_ds)

    @property
    def testing_df(self, parallel=False):
        """Returns the testing DataFrame.
        Returns:
            pd.DataFrame: Testing data DataFrame
        """
        return self.df_from_ds(self.testing_ds)


class DataCreator:
    """Generates tfrecords files from ROOT map files.
    """

    def __init__(self, directory, geom, split, join, parallel, all_maps):
        """Initialise the DataCreator.
        Args:
            directory (str): Input production directory
            geom (str): Geometry to use
            split (float): Validation and testing fractional data split
            join (int): Number of input files to combine together
            parallel (bool): Should we run parallel processes?
            all_maps (bool): Should we generate all the maps?
        """
        self.split = split
        self.join = join
        self.parallel = parallel
        self.all_maps = all_maps
        self.init(directory, geom)

    def init(self, directory, geom):
        """Initialise the output directories.
        Args:
            directory (str): Production input/output directory path
            geom (str): CHIPS geometry to use
        """
        self.in_dir = os.path.join(directory, "map/", geom)
        self.out_dir = os.path.join(directory, "tf/", geom)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "train/"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "val/"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "test/"), exist_ok=True)

    def bytes_feature(self, value):
        """Returns a BytesList feature from a string/byte.
        Args:
            value (str): Raw string format of an array
        Returns:
            tf.train.Feature: A BytesList feature
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def gen_examples(self, true, reco):
        """Generates a list of examples from the input .root map file.
        Args:
            true (uproot TTree): True TTree from input file
            reco (uproot TTree): Reco TTree from input file
        Returns:
            List[tf.train.Example]: List of examples
        """
        # Get the numpy arrays from the .root map file, we need to seperate by type
        # for the deserialisation during reading to work correctly.
        true_pars_i = np.stack((  # True Parameters (integers)
            true.array('t_nu'),
            true.array('t_code')),
            axis=1)
        true_pars_f = np.stack((  # True Parameters (floats)
            true.array('t_vtxX'),
            true.array('t_vtxY'),
            true.array('t_vtxZ'),
            true.array('t_vtxT'),
            true.array('t_nuEnergy')),
            axis=1)
        true_prim_i = true.array('t_p_pdgs')  # True Primaries (integers)
        true_prim_f = np.stack((  # True Primaries (floats)
            true.array('t_p_energies'),
            true.array('t_p_dirTheta'),
            true.array('t_p_dirPhi')),
            axis=1)
        reco_pars_i = np.stack((  # Reco Parameters (integers)
            reco.array('r_raw_num_hits'),
            reco.array('r_filtered_num_hits'),
            reco.array('r_num_hough_rings')),
            axis=1)
        reco_pars_f = np.stack((  # Reco Parameters (floats)
            reco.array('r_raw_total_digi_q'),
            reco.array('r_filtered_total_digi_q'),
            reco.array('r_first_ring_height'),
            reco.array('r_last_ring_height'),
            reco.array('r_vtxX'),
            reco.array('r_vtxY'),
            reco.array('r_vtxZ'),
            reco.array('r_vtxT'),
            reco.array('r_dirTheta'),
            reco.array('r_dirPhi')),
            axis=1)

        channels = []
        ranges = []

        channels.append('r_raw_charge_map_vtx')
        ranges.append((0.0, 15.0))
        channels.append('r_raw_time_map_vtx')
        ranges.append((0.0, 80.0))
        channels.append('r_filtered_hit_hough_map_vtx')
        ranges.append((0.0, 1500.0))

        if self.all_maps:
            channels.append('r_raw_hit_map_origin')
            ranges.append((0.0, 15.0))
            channels.append('r_raw_charge_map_origin')
            ranges.append((0.0, 15.0))
            channels.append('r_raw_time_map_origin')
            ranges.append((0.0, 80.0))
            channels.append('r_filtered_hit_map_origin')
            ranges.append((0.0, 15.0))
            channels.append('r_filtered_charge_map_origin')
            ranges.append((0.0, 15.0))
            channels.append('r_filtered_time_map_origin')
            ranges.append((0.0, 80.0))
            channels.append('r_raw_hit_map_vtx')
            ranges.append((0.0, 15.0))
            channels.append('r_filtered_hit_map_vtx')
            ranges.append((0.0, 15.0))
            channels.append('r_filtered_charge_map_vtx')
            ranges.append((0.0, 15.0))
            channels.append('r_filtered_time_map_vtx')
            ranges.append((0.0, 80.0))

        channel_images = []
        for i, channel in enumerate(channels):
            channel_images.append(reco.array(channel))

        image = np.stack(channel_images, axis=3)

        examples = []  # Generate examples using a feature dict
        for i in range(len(true_pars_i)):
            feature_dict = {
                'true_pars_i': self.bytes_feature(true_pars_i[i].tostring()),
                'true_pars_f': self.bytes_feature(true_pars_f[i].tostring()),
                'true_prim_i': self.bytes_feature(true_prim_i[i].tostring()),
                'true_prim_f': self.bytes_feature(true_prim_f[i].tostring()),
                'reco_pars_i': self.bytes_feature(reco_pars_i[i].tostring()),
                'reco_pars_f': self.bytes_feature(reco_pars_f[i].tostring()),
                'image': self.bytes_feature(image[i].tostring())
            }
            examples.append(tf.train.Example(features=tf.train.Features(feature=feature_dict)))

        return examples

    def write_examples(self, name, examples):
        """Write a list of examples to a tfrecords file.
        Args:
            name (str): Output .tfrecords file path
            examples (List[tf.train.Example]): List of examples
        """
        with tf.io.TFRecordWriter(name) as writer:
            for example in examples:
                writer.write(example.SerializeToString())

    def preprocess_files(self, num, files):
        """Preprocess joined .root map files into train, val and test tfrecords files.
        Args:
            num (int): Job number
            files (list[str]): List of input files to use
        """
        print('Processing job {}...'.format(num))
        examples = []
        for file in files:
            file_u = uproot.open(file)
            try:
                examples.extend(self.gen_examples(file_u['true'], file_u['reco']))
            except Exception as err:  # Catch when there is an uproot exception and skip
                print('Error:', type(err), err)
                pass

        # Split into training, validation and testing samples
        random.shuffle(examples)  # Shuffle the examples list
        val_split = int((1.0-self.split-self.split) * len(examples))
        test_split = int((1.0-self.split) * len(examples))
        train_examples = examples[:val_split]
        val_examples = examples[val_split:test_split]
        test_examples = examples[test_split:]

        self.write_examples(
            os.path.join(self.out_dir, 'train/', str(num) + '_train.tfrecords'), train_examples)
        self.write_examples(
            os.path.join(self.out_dir, 'val/', str(num) + '_val.tfrecords'), val_examples)
        self.write_examples(
            os.path.join(self.out_dir, 'test/', str(num) + '_test.tfrecords'), test_examples)

    def run(self):
        """Preprocess all the files from the input directory into tfrecords.
        """
        files = [os.path.join(self.in_dir, file) for file in os.listdir(self.in_dir)]
        file_lists = [files[n:n+self.join] for n in range(0, len(files), self.join)]
        if self.parallel:  # File independence allows for parallelisation
            Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(
                self.preprocess_files)(counter, f_list) for counter, f_list in enumerate(
                    file_lists))
        else:
            for counter, f_list in enumerate(file_lists):
                self.preprocess_files(counter, f_list)
