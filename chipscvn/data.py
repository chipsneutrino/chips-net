"""Data creation and loading module

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module contains both the DataCreator and DataLoader classes, these
are used to firstly generate tfrecords files from ROOT hitmap files and
then to read these on the fly using tf.datasets at model training or
evaluation.
"""

import os
from joblib import Parallel, delayed
import multiprocessing
import random

import uproot
import numpy as np
import tensorflow as tf


class DataLoader:
    """Generates tf datasets from the configuration."""
    def __init__(self, config):
        self.config = config
        self.init()

    def init(self):
        """Initialise the PDG, type and category lookup tables and input directories."""

        # Map nuel and numu (Total = 2)
        # 0 = Nuel neutrino
        # 1 = Numu neutrino (cosmic muons are included in this)
        nu_keys = tf.constant([11, 12, 13, 14])
        nu_vals = tf.constant([0,  0,  1,  1])
        self.nu_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(nu_keys, nu_vals), -1)

        # Map interaction types (Total = 10)
        # 0 = CC-QEL
        # 1 = CC-RES
        # 2 = CC-DIS
        # 3 = CC-COH
        # 4 = NC-QEL
        # 5 = NC-RES
        # 6 = NC-DIS
        # 7 = NC-COH
        # 8 = Cosmic
        # 9 = Other
        int_keys = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 91, 92, 96, 97, 98, 99, 100])
        int_vals = tf.constant([9, 0, 4, 1, 1, 1, 5, 5, 5, 5,  2,  6,  7,  3,  9,  9,   8])
        self.int_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(int_keys, int_vals), -1)

        # Map to all categories (Total = 19)
        # Category keys are a string of pdg+type, e.g an nuel ccqe event is '0'+'0' = '00'
        # 0 = Nuel CC-QEL
        # 1 = Nuel CC-RES
        # 2 = Nuel CC-DIS
        # 3 = Nuel CC-COH
        # 4 = Numu CC-QEL
        # 5 = Numu CC-RES
        # 6 = Numu CC-DIS
        # 7 = Numu CC-COH
        # 8 = Nuel NC-QEL
        # 9 = Nuel NC-RES
        # 10 = Nuel NC-DIS
        # 11 = Nuel NC-COH
        # 12 = Numu NC-QEL
        # 13 = Numu NC-RES
        # 14 = Numu NC-DIS
        # 15 = Numu NC-COH
        # 16 = Cosmic
        # 17 = Nuel Other
        # 18 = Numu Other
        cat_keys = tf.constant(['00', '01', '02', '03', '10', '11', '12', '13',
                                '04', '05', '06', '07', '14', '15', '16', '17',
                                '18', '09', '19'], dtype=tf.string)
        cat_vals = tf.constant([0, 1, 2, 3, 4, 5, 6, 7,
                                8, 9, 10, 11, 12, 13, 14, 15,
                                16, 17, 18])
        self.cat_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(cat_keys, cat_vals), -1)

        # The following mappings are used to generate the inputs to different model types

        # Map a cosmic flag (Total = 2)
        # 0 = Not a Cosmic
        # 1 = A Cosmic
        cosmic_keys = tf.constant([0, 1, 2, 3, 4, 5, 6, 7,
                                   8, 9, 10, 11, 12, 13, 14, 15,
                                   16, 17, 18])
        cosmic_vals = tf.constant([0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0,
                                   1, 0, 0])
        self.cosmic_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(cosmic_keys, cosmic_vals), -1)

        # Map to full_combined categories (Total = 5)
        # 0 = Nuel CC
        # 1 = Numu CC
        # 2 = NC
        # 3 = Cosmic
        # 4 = Other
        comb_keys = tf.constant([0, 1, 2, 3, 4, 5, 6, 7,
                                 8, 9, 10, 11, 12, 13, 14, 15,
                                 16, 17, 18])
        comb_vals = tf.constant([0, 0, 0, 0, 1, 1, 1, 1,
                                 2, 2, 2, 2, 2, 2, 2, 2,
                                 3, 4, 4])
        self.comb_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(comb_keys, comb_vals), -1)

        # Map to nc_nu_combined categories (Total = 14)
        # 0 = Nuel CC-QEL
        # 1 = Nuel CC-RES
        # 2 = Nuel CC-DIS
        # 3 = Nuel CC-COH
        # 4 = Numu CC-QEL
        # 5 = Numu CC-RES
        # 6 = Numu CC-DIS
        # 7 = Numu CC-COH
        # 8 = NC-QEL
        # 9 = NC-RES
        # 10 = NC-DIS
        # 11 = NC-COH
        # 12 = Cosmic
        # 13 = Other
        nu_nc_comb_keys = tf.constant([0, 1, 2, 3, 4, 5, 6, 7,
                                       8, 9, 10, 11, 12, 13, 14, 15,
                                       16, 17, 18])
        nu_nc_comb_vals = tf.constant([0, 1, 2, 3, 4, 5, 6, 7,
                                       8, 9, 10, 11, 8, 9, 10, 11,
                                       12, 13, 13])
        self.nu_nc_comb_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(nu_nc_comb_keys, nu_nc_comb_vals), -1)

        # Map to nc_combined categories (Total = 11)
        # 0 = Nuel CC-QEL
        # 1 = Nuel CC-RES
        # 2 = Nuel CC-DIS
        # 3 = Nuel CC-COH
        # 4 = Numu CC-QEL
        # 5 = Numu CC-RES
        # 6 = Numu CC-DIS
        # 7 = Numu CC-COH
        # 8 = NC
        # 9 = Cosmic
        # 10 = Other
        nc_comb_keys = tf.constant([0, 1, 2, 3, 4, 5, 6, 7,
                                    8, 9, 10, 11, 12, 13, 14, 15,
                                    16, 17, 18])
        nc_comb_vals = tf.constant([0, 1, 2, 3, 4, 5, 6, 7,
                                    8, 8, 8, 8, 8, 8, 8, 8,
                                    9, 10, 10])
        self.nc_comb_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(nc_comb_keys, nc_comb_vals), -1)

        # Generate the lists of train, val and test file directories from the configuration
        self.train_dirs = [os.path.join(in_dir, 'train') for in_dir in self.config.data.input_dirs]
        self.val_dirs = [os.path.join(in_dir, 'val') for in_dir in self.config.data.input_dirs]
        self.test_dirs = [os.path.join(in_dir, 'test') for in_dir in self.config.data.input_dirs]

    def parse(self, serialised_example):
        """Parses a single serialised example into an input plus a labels dict."""
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

        # Decode the parameter arrays using their types
        true_pars_i = tf.io.decode_raw(example['true_pars_i'], tf.int32)
        true_pars_f = tf.io.decode_raw(example['true_pars_f'], tf.float32)
        true_prim_i = tf.io.decode_raw(example['true_prim_i'], tf.int32)
        true_prim_f = tf.io.decode_raw(example['true_prim_f'], tf.float32)
        reco_pars_i = tf.io.decode_raw(example['reco_pars_i'], tf.int32)
        reco_pars_f = tf.io.decode_raw(example['reco_pars_f'], tf.float32)

        # Do all the base mapping using the lookup tables
        pdg = self.nu_table.lookup(true_pars_i[0])
        type = self.int_table.lookup(true_pars_i[1])
        category = self.cat_table.lookup(
            tf.strings.join((tf.strings.as_string(pdg), tf.strings.as_string(type))))

        # Do all the model specific mapping using the lookup tables
        cosmic = self.cosmic_table.lookup(category)
        full_comb = self.comb_table.lookup(category)
        nu_nc_comb = self.nu_nc_comb_table.lookup(category)
        nc_comb = self.nc_comb_table.lookup(category)

        # Need to reshape the primary particle array
        true_prim_f = tf.reshape(true_prim_f, [3, 10])

        labels = {  # We generate a dictionary with all the true labels
            't_nu': pdg,
            't_code': type,
            't_cat': category,
            't_cosmic_cat': cosmic,
            't_full_cat': full_comb,
            't_nu_nc_cat': nu_nc_comb,
            't_nc_cat': nc_comb,
            't_vtxX': true_pars_f[0],
            't_vtxY': true_pars_f[1],
            't_vtxZ': true_pars_f[2],
            't_vtxT': true_pars_f[3],
            't_nuEnergy': true_pars_f[4],
            't_p_pdgs': true_prim_i,
            't_p_energies': true_prim_f[0],
            't_p_dirTheta': true_prim_f[1],
            't_p_dirPhi': true_prim_f[2]
        }

        inputs = {  # We generate a dictionary with the images and other input parameters
            'r_raw_num_hits': reco_pars_i[0],
            'r_filtered_num_hits': reco_pars_i[1],
            'r_num_hough_rings': reco_pars_i[2],
            'r_raw_total_digi_q': reco_pars_f[0],
            'r_filtered_total_digi_q': reco_pars_f[1],
            'r_first_ring_height': reco_pars_f[2],
            'r_last_ring_height': reco_pars_f[3],
            'r_vtxX': tf.math.divide(reco_pars_f[4], self.config.data.par_scale[0]),
            'r_vtxY': tf.math.divide(reco_pars_f[5], self.config.data.par_scale[1]),
            'r_vtxZ': tf.math.divide(reco_pars_f[6], self.config.data.par_scale[2]),
            'r_vtxT': reco_pars_f[7],
            'r_dirTheta': tf.math.divide(reco_pars_f[8], self.config.data.par_scale[3]),
            'r_dirPhi': tf.math.divide(reco_pars_f[9], self.config.data.par_scale[4])
        }

        # Decode and reshape the "image" into a tf tensor
        full_image = tf.io.decode_raw(example['image'], tf.uint8)
        if self.config.data.all_chan:
            full_image = tf.reshape(full_image, [64, 64, 13])
        else:
            full_image = tf.reshape(full_image, [64, 64, 3])

        # 'unstack' the image and manipulate each channel individually
        unstacked = tf.unstack(full_image, axis=2)
        channels = []
        for i, enabled in enumerate(self.config.data.channels):
            if enabled:
                # Cast to float at tf does this anyway, and scale to [0,1]
                unstacked[i] = tf.cast(unstacked[i], tf.float32)
                unstacked[i] = unstacked[i] / 256.0

                # Apply a random distribution to the channel
                rand = tf.random.normal(shape=[64, 64], mean=1,
                                        stddev=self.config.data.rand[i],
                                        dtype=tf.float32)
                unstacked[i] = tf.math.multiply(unstacked[i], rand)

                # Apply a shift to the channel
                shift = tf.fill([64, 64], (1.0 + self.config.data.shift[i]))
                unstacked[i] = tf.math.multiply(unstacked[i], shift)

                # TODO: Could take values below zero, change to prevent this
                channels.append(unstacked[i])

        # Choose to either stack the channels back into a single tensor or keep them seperate
        if self.config.data.stack:
            image = tf.stack(channels, axis=2)
            inputs['image_0'] = image
        else:
            for i, input_image in enumerate(channels):
                input_image = tf.expand_dims(input_image, 2)
                input_name = 'image_' + str(i)
                inputs[input_name] = input_image

        return inputs, labels

    def filter_other(self, inputs, labels):
        """Filters out 'other' cateogory events."""
        if (labels['t_cat']) == 17 or (labels['t_cat'] == 18):
            return False
        else:
            return True

    def dataset(self, dirs):
        """Returns a dataset formed from all the files in the input directories."""
        files = []  # Add all files in dirs to a list
        for d in dirs:
            for file in os.listdir(d):
                files.append(os.path.join(d, file))

        random.shuffle(files)  # Shuffle the list as an additionally randomisation to "interleave"
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.interleave(tf.data.TFRecordDataset,
                           cycle_length=len(files),
                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        ds = ds.map(lambda x: self.parse(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.filter(self.filter_other)
        return ds

    def train_data(self):
        """Returns the training dataset."""
        ds = self.dataset(self.train_dirs)
        ds = ds.batch(self.config.data.batch_size, drop_remainder=True)
        ds = ds.take(self.config.data.train_examples)
        return ds

    def val_data(self):
        """Returns the validation dataset."""
        ds = self.dataset(self.val_dirs)
        ds = ds.batch(self.config.data.batch_size, drop_remainder=True)
        ds = ds.take(int(self.config.data.val_examples))
        return ds

    def test_data(self):
        """Returns the testing dataset."""
        ds = self.dataset(self.test_dirs)
        ds = ds.batch(self.config.data.batch_size, drop_remainder=True)
        ds = ds.take(self.config.data.test_examples)
        return ds


class DataCreator:
    """Generates tfrecord files from ROOT map files."""
    def __init__(self, directory, geom, split, join, single, all_maps):
        self.split = split
        self.join = join
        self.single = single
        self.all_maps = all_maps
        self.init(directory, geom)

    def init(self, directory, geom):
        """Initialise the output directories."""
        self.in_dir = os.path.join(directory, "map/", geom)
        self.out_dir = os.path.join(directory, "tf/", geom)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "train/"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "val/"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "test/"), exist_ok=True)

    def bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def gen_examples(self, true, reco):
        """Generates a list of examples from the input .root map file."""

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
        """Write a list of examples to a tfrecords file."""
        with tf.io.TFRecordWriter(name) as writer:
            for example in examples:
                writer.write(example.SerializeToString())

    def preprocess_file(self, num, files):
        """Preprocess joined .root map files into train, val and test tfrecords files."""
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

    def preprocess(self):
        """Preprocess all the files from the input directory into tfrecords."""
        files = [os.path.join(self.in_dir, file) for file in os.listdir(self.in_dir)]
        file_lists = [files[n:n+self.join] for n in range(0, len(files), self.join)]
        if not self.single:  # File independence allows for parallelisation
            Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(
                self.preprocess_file)(counter, f_list) for counter, f_list in enumerate(file_lists))
        else:  # For debugging we keep the option to use a single process
            for counter, f_list in enumerate(file_lists):
                self.preprocess_file(counter, f_list)
