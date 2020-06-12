# -*- coding: utf-8 -*-

"""Data creation and reading module

This module contains both the Creator and Reader classes, these
are used to firstly generate tfrecords files from ROOT hitmap files and
then to read these on the fly using tf.datasets at model training or
evaluation.
"""

import os
from joblib import Parallel, delayed
import multiprocessing
import random
import math

import pandas as pd
import uproot
import numpy as np
import tensorflow as tf
from dotmap import DotMap
from particle import Particle


class Reader:
    """Generates tf datasets for training/evaluation from the configuration.
    These can be read on the fly by Tensorflow so that the entire dataset does
    not need to be loaded into memory.
    """

    def __init__(self, config):
        """Initialise the Reader.
        Args:
            config (str): Dotmap configuration namespace
        """
        self.config = config
        self.train_dirs = [os.path.join(in_dir, 'train') for in_dir in config.data.input_dirs]
        self.val_dirs = [os.path.join(in_dir, 'val') for in_dir in config.data.input_dirs]
        self.test_dirs = [os.path.join(in_dir, 'test') for in_dir in config.data.input_dirs]

        self.image_shape = [
            self.config.data.img_size[0],
            self.config.data.img_size[1],
            len(self.config.data.channels)
        ]

    @tf.function
    def parse(self, serialised_example):
        """Parses a single serialised example into both an input and labels dict.
        Args:
            serialised_example (tf.Example): A single example from .tfrecords file
        Returns:
            Tuple[dict, dict]: (Inputs dictionary, Labels dictionary)
        """
        features = {
            'inputs_image': tf.io.FixedLenFeature([], tf.string),
            'inputs_other': tf.io.FixedLenFeature([], tf.string),
            'labels_i': tf.io.FixedLenFeature([], tf.string),
            'labels_f': tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(serialised_example, features)

        inputs, labels = {}, {}  # The two dictionaries to fill

        # Decode and reshape the 'image' into a tf tensor, reshape, then cast correctly and scale
        image = tf.io.decode_raw(example['inputs_image'], tf.uint8)
        image = tf.reshape(image, self.image_shape)
        image = tf.cast(image, tf.float32) / 256.0  # Cast to float and salce to [0,1]

        # Check if we actually need to unstack the image before we do...
        unstacked = tf.unstack(image, axis=2)
        channels = []
        for i, enabled in enumerate(self.config.data.channels):
            if enabled:
                if self.config.data.augment:
                    rand_shift = tf.random.normal(
                        shape=self.config.data.img_size,
                        mean=(1.0 + self.config.data.shift[i]),
                        stddev=self.config.data.rand[i],
                        dtype=tf.float32
                    )
                    unstacked[i] = tf.math.multiply(unstacked[i], rand_shift)
                channels.append(unstacked[i])

        # Choose to either stack the channels back into a single tensor or keep them seperate
        if self.config.data.unstack:
            for i, input_image in enumerate(channels):
                inputs['image_'+str(i)] = tf.expand_dims(input_image, 2)
        else:
            inputs['image_0'] = tf.stack(channels, axis=2)

        # Decode the other inputs and append to inputs dictionary
        inputs_other = tf.io.decode_raw(example['inputs_other'], tf.float32)
        inputs['r_raw_total_digi_q'] = inputs_other[0]
        inputs['r_first_ring_height'] = inputs_other[1]
        inputs['r_vtxX'] = inputs_other[2]
        inputs['r_vtxY'] = inputs_other[3]
        inputs['r_vtxZ'] = inputs_other[4]
        inputs['r_dirTheta'] = inputs_other[5]
        inputs['r_dirPhi'] = inputs_other[6]

        # Decode integer labels and append to labels dictionary
        labels_i = tf.io.decode_raw(example['labels_i'], tf.int32)
        labels[MAP_NU_TYPE.name] = labels_i[0]
        labels[MAP_SIGN_TYPE.name] = labels_i[1]
        labels[MAP_INT_TYPE.name] = labels_i[2]
        labels[MAP_ALL_CAT.name] = labels_i[3]
        labels[MAP_COSMIC_CAT.name] = labels_i[4]
        labels[MAP_FULL_COMB_CAT.name] = labels_i[5]
        labels[MAP_NU_NC_COMB_CAT.name] = labels_i[6]
        labels[MAP_NC_COMB_CAT.name] = labels_i[7]
        labels["prim_total"] = labels_i[8]
        labels["prim_p"] = labels_i[9]
        labels["prim_cp"] = labels_i[10]
        labels["prim_np"] = labels_i[11]
        labels["prim_g"] = labels_i[12]

        # Decode float labels and append to the labels dictionary
        labels_f = tf.io.decode_raw(example['labels_f'], tf.float32)
        labels['t_vtxX'] = labels_f[0]
        labels['t_vtxY'] = labels_f[1]
        labels['t_vtxZ'] = labels_f[2]
        labels['t_nuEnergy'] = labels_f[3]

        # Append labels to inputs if needed for multitask network
        for label in self.config.model.labels:
            inputs["input_"+label] = labels[label]

        return inputs, labels

    @tf.function
    def strip(self, inputs, labels):
        """Strips all labels except those needed in training/validation.
        Args:
            Tuple[dict, dict]: (Inputs dictionary, Labels dictionary)
        Returns:
            Tuple[dict, dict]: (Inputs dictionary, Stripped labels dictionary)
        """
        labels = {k: labels[k] for k in self.config.model.labels}
        return inputs, labels

    def filter_cats(self, inputs, labels):
        return tf.math.equal(labels['t_all_cat'], self.config.data.cat_select)

    def dataset(self, dirs, strip=True):
        """Returns a dataset formed from all the files in the input directories.
        Args:
            dirs (list[str]): List of input directories
        Returns:
            tf.dataset: The generated dataset
        """
        # Generate list of input files and shuffle
        files = []
        for d in dirs:
            for file in os.listdir(d):
                files.append(os.path.join(d, file))
        random.seed(8)
        random.shuffle(files)

        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.interleave(
            tf.data.TFRecordDataset,
            cycle_length=len(files),
            block_length=1,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        ds = ds.map(lambda x: self.parse(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if strip:
            ds = ds.map(lambda x, y: self.strip(x, y),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.config.data.cat_select != -1:
            ds = ds.filter(self.filter_cats)

        return ds

    def df_from_ds(self, ds):
        """Create a pandas dataframe from a tf dataset
        Args:
            ds (tf.dataset): Input dataset
            num_events (int): Number of events to include
        Returns:
            pandas.DataFrame: DataFrame generated from the dataset
        """
        events = {}
        for x, y in ds:
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

    def training_ds(self, num_events, batch_size=None, strip=True):
        """Returns the training dataset.
        Returns:
            tf.dataset: The training dataset
        """
        ds = self.dataset(self.train_dirs, strip=strip)
        ds = ds.take(num_events)
        if batch_size is not None:
            ds = ds.batch(batch_size, drop_remainder=True)
        return ds

    def validation_ds(self, num_events, batch_size=None, strip=True):
        """Returns the validation dataset.
        Returns:
            tf.dataset: The validation dataset
        """
        ds = self.dataset(self.val_dirs, strip=strip)
        ds = ds.take(num_events)
        if batch_size is not None:
            ds = ds.batch(batch_size, drop_remainder=True)
        return ds

    def testing_ds(self, num_events, batch_size=None, strip=False):
        """Returns the testing dataset.
        Returns:
            tf.dataset: The testing dataset
        """
        ds = self.dataset(self.test_dirs, strip=strip)
        ds = ds.take(num_events)
        if batch_size is not None:
            ds = ds.batch(batch_size, drop_remainder=True)
        return ds

    def training_df(self, num_events):
        """Returns the training DataFrame.
        Args:
            num_events (int): Number of events to include
        Returns:
            pd.DataFrame: Training data DataFrame
        """
        return self.df_from_ds(self.training_ds(num_events, 64))

    def validation_df(self, num_events):
        """Returns the validation DataFrame.
        Args:
            num_events (int): Number of events to include
        Returns:
            pd.DataFrame: Validation data DataFrame
        """
        return self.df_from_ds(self.validation_ds(num_events, 64))

    def testing_df(self, num_events):
        """Returns the testing DataFrame.
        Args:
            num_events (int): Number of events to include
        Returns:
            pd.DataFrame: Testing data DataFrame
        """
        return self.df_from_ds(self.testing_ds(num_events, 64))


class Creator:
    """Generates tfrecords files from ROOT map files.
    """

    def __init__(self, config):
        """Initialise the Creator.
        Args:
            config (str): Dotmap configuration namespace
        """
        self.config = config
        os.makedirs(config.create.out_dir, exist_ok=True)
        os.makedirs(os.path.join(config.create.out_dir, "train/"), exist_ok=True)
        os.makedirs(os.path.join(config.create.out_dir, "val/"), exist_ok=True)
        os.makedirs(os.path.join(config.create.out_dir, "test/"), exist_ok=True)

    def bytes_feature(self, value):
        """Returns a BytesList feature from a string/byte.
        Args:
            value (str): Raw string format of an array
        Returns:
            tf.train.Feature: A BytesList feature
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def count_primaries(self, pdgs, energies):
        """Counts the number of Cherenkov threshold passing primaries for each type in the event.

        We count the number of particles for...
            - protons (above the cherenkov threshold)
            - charged pions (above the cherenkov threshold)
            - neutral pions (above a few pair productions in energy)
            - photons (above a few pair productions in energy)
            - total (total above the their threshold)
        in the event, either (0, 1, 2, n). Anything above 2 is classified into the 'n' category

        Args:
            pdgs (np.array): Primary particle pdgs
            energies (np.array): Primary particle energies
        Returns:
            np.array: array of the particle counts
        """
        events = []
        for ev_pdgs, ev_energies in zip(pdgs, energies):  # loop through all events
            counts = np.array([0, 0, 0, 0, 0])
            for i, pdg in enumerate(ev_pdgs):
                if pdg == -999:
                    continue
                elif pdg in [2212, 2212] and ev_energies[i] > PROTON_THRESHOLD:
                    counts[0] += 1
                    counts[1] += 1
                elif pdg in [211, -211] and ev_energies[i] > CP_THRESHOLD:
                    counts[0] += 1
                    counts[2] += 1
                elif pdg in [111] and ev_energies[i] > NP_THRESHOLD:
                    counts[0] += 1
                    counts[3] += 1
                elif pdg in [22] and ev_energies[i] > GAMMA_THRESHOLD:
                    counts[0] += 1
                    counts[4] += 1
            counts = np.clip(counts, a_min=0, a_max=3)  # 4th value is for n>2 particles
            events.append(counts)
        return np.stack(events, axis=0)

    def gen_examples(self, true, reco):
        """Generates a list of examples from the input .root map file.
        Args:
            true (uproot TTree): True TTree from input file
            reco (uproot TTree): Reco TTree from input file
        Returns:
            List[tf.train.Example]: List of examples
        """
        # First setup the input image
        channels = ['r_raw_charge_map_vtx', 'r_raw_time_map_vtx', 'r_raw_hit_hough_map_vtx']
        if self.config.create.all_maps:
            channels.append('r_raw_hit_map_origin')
            channels.append('r_raw_charge_map_origin')
            channels.append('r_raw_time_map_origin')
            channels.append('r_filtered_hit_map_origin')
            channels.append('r_filtered_charge_map_origin')
            channels.append('r_filtered_time_map_origin')
            channels.append('r_raw_hit_map_vtx')
            channels.append('r_filtered_hit_map_vtx')
            channels.append('r_filtered_charge_map_vtx')
            channels.append('r_filtered_time_map_vtx')
            channels.append('r_raw_hit_map_iso')
            channels.append('r_raw_charge_map_iso')
            channels.append('r_raw_time_map_iso')
            channels.append('r_filtered_hit_map_iso')
            channels.append('r_filtered_charge_map_iso')
            channels.append('r_filtered_time_map_iso')
        channel_images = [reco.array(channel) for channel in channels]
        inputs_image = np.stack(channel_images, axis=3)

        # Next setup the other inputs, mainly reconstructed variables
        inputs_other = np.stack((  # Reco Parameters (floats)
            reco.array('r_raw_total_digi_q'),
            reco.array('r_first_ring_height'),
            reco.array('r_vtxX')/self.config.create.par_scale[0],
            reco.array('r_vtxY')/self.config.create.par_scale[1],
            reco.array('r_vtxZ')/self.config.create.par_scale[2],
            reco.array('r_dirTheta')/self.config.create.par_scale[3],
            reco.array('r_dirPhi')/self.config.create.par_scale[4]),
            axis=1)

        # Next setup the integer labels, we need to map to the different categories etc...
        n_arr = np.vectorize(MAP_NU_TYPE.table.get)(true.array('t_nu'))
        s_arr = np.vectorize(MAP_SIGN_TYPE.table.get)(true.array('t_nu'))
        i_arr = np.vectorize(MAP_INT_TYPE.table.get)(true.array('t_code'))
        cat_arr = np.array([MAP_ALL_CAT.table[(n, i)] for n, i in zip(n_arr, i_arr)])
        cosmic_arr = np.array([MAP_COSMIC_CAT.table[(n, i)] for n, i in zip(n_arr, i_arr)])
        comb_arr = np.array([MAP_FULL_COMB_CAT.table[(n, i)] for n, i in zip(n_arr, i_arr)])
        nu_nc_comb_arr = np.array([MAP_NU_NC_COMB_CAT.table[(n, i)] for n, i in zip(n_arr, i_arr)])
        nc_comb_arr = np.array([MAP_NC_COMB_CAT.table[(n, i)] for n, i in zip(n_arr, i_arr)])

        counts = self.count_primaries(true.array('t_p_pdgs'), true.array('t_p_energies'))

        labels_i = np.stack((  # True Parameters (integers)
            n_arr,
            s_arr,
            i_arr,
            cat_arr,
            cosmic_arr,
            comb_arr,
            nu_nc_comb_arr,
            nc_comb_arr,
            counts[:, 0],
            counts[:, 1],
            counts[:, 2],
            counts[:, 3],
            counts[:, 4]),
            axis=1).astype(np.int32)

        labels_f = np.stack((  # True Parameters (floats)
            true.array('t_vtxX'),
            true.array('t_vtxY'),
            true.array('t_vtxZ'),
            true.array('t_nuEnergy')),
            axis=1)

        examples = []  # Generate examples using a feature dict
        for i in range(len(labels_i)):
            feature_dict = {
                'inputs_image': self.bytes_feature(inputs_image[i].tostring()),
                'inputs_other': self.bytes_feature(inputs_other[i].tostring()),
                'labels_i': self.bytes_feature(labels_i[i].tostring()),
                'labels_f': self.bytes_feature(labels_f[i].tostring())
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
        examples = []
        print("job {}...".format(num))
        for file in files:
            file_u = uproot.open(file)
            examples.extend(self.gen_examples(file_u['true'], file_u['reco']))

        # Split into training, validation and testing samples
        random.shuffle(examples)  # Shuffle the examples list
        val_split = int((1.0-self.config.create.split-self.config.create.split) * len(examples))
        test_split = int((1.0-self.config.create.split) * len(examples))
        train_examples = examples[:val_split]
        val_examples = examples[val_split:test_split]
        test_examples = examples[test_split:]

        self.write_examples(
            os.path.join(self.config.create.out_dir, 'train/', str(num) + '_train.tfrecords'),
            train_examples)
        self.write_examples(
            os.path.join(self.config.create.out_dir, 'val/', str(num) + '_val.tfrecords'),
            val_examples)
        self.write_examples(
            os.path.join(self.config.create.out_dir, 'test/', str(num) + '_test.tfrecords'),
            test_examples)

    def run(self):
        """Preprocess all the files from the input directory into tfrecords.
        """
        files = []
        for directory in self.config.create.input_dirs:
            files.extend([os.path.join(directory, file) for file in os.listdir(directory)])

        random.seed(8)
        random.shuffle(files)  # Shuffle the file list

        file_lists = [files[n:n+self.config.create.join] for n in range(
            0, len(files), self.config.create.join)]
        if self.config.create.parallel:  # File independence allows for parallelisation
            Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(
                self.preprocess_files)(counter, f_list) for counter, f_list in enumerate(
                    file_lists))
        else:
            for counter, f_list in enumerate(file_lists):
                self.preprocess_files(counter, f_list)


"""Declare constants for use in primary particle counting"""
INDEX = 1.344  # for 405nm at ~4 degrees celcius
PROTON_THRESHOLD = math.sqrt(math.pow(Particle.from_pdgid(2212).mass, 2)/(1-(1/math.pow(INDEX, 2))))
CP_THRESHOLD = math.sqrt(math.pow(Particle.from_pdgid(211).mass, 2)/(1-(1/math.pow(INDEX, 2))))
NP_THRESHOLD = 20 * Particle.from_pdgid(11).mass
GAMMA_THRESHOLD = 20 * Particle.from_pdgid(11).mass


def get_map(name):
    """Getting category mapping dict from name.
    Args:
        name (str): Name of mapping
    Returns:
        dict: Mapping dictionary
    """
    for map in [MAP_NU_TYPE, MAP_INT_TYPE, MAP_ALL_CAT,
                MAP_COSMIC_CAT, MAP_FULL_COMB_CAT, MAP_NU_NC_COMB_CAT,
                MAP_NC_COMB_CAT]:
        if map.name == name:
            return map
    return None


"""Map to electron or muon types (Total = 2) (cosmic muons are included in this)"""
MAP_NU_TYPE = DotMap({
    'name': 't_nu_type',
    'categories': 1,
    'labels': [
        'Nuel',         # 0
        'Numu'],        # 1
    'table': {
        11: 0,          # el-
        -11: 0,         # el+
        12: 0,          # el neutrino
        -12: 0,         # el anti neutrino
        13: 1,          # mu-
        -13: 1,         # mu+
        14: 1,          # mu neutrino
        -14: 1}         # mu anti neutrino
})


"""Map to particle vs anti-particle (Total = 2) (cosmic muons are included in this)"""
MAP_SIGN_TYPE = DotMap({
    'name': 't_sign_type',
    'categories': 1,
    'labels': [
        'Nu',           # 0
        'Anu'],         # 1
    'table': {
        11: 0,          # el-
        -11: 1,         # el+
        12: 0,          # el neutrino
        -12: 1,         # el anti neutrino
        13: 0,          # mu-
        -13: 1,         # mu+
        14: 0,          # mu neutrino
        -14: 1}         # mu anti neutrino
})


"""Map interaction types (Total = 13)
We put IMD, ElasticScattering and InverseMuDecay into 'NC-OTHER' for simplicity"""
MAP_INT_TYPE = DotMap({
    'name': 't_int_type',
    'categories': 12,
    'labels': [
        'CC-QEL',       # 0
        'CC-RES',       # 1
        'CC-DIS',       # 2
        'CC-COH',       # 3
        'CC-MEC',       # 4
        'CC-OTHER',     # 5
        'NC-QEL',       # 6
        'NC-RES',       # 7
        'NC-DIS',       # 8
        'NC-COH',       # 9
        'NC-MEC',       # 10
        'NC-OTHER',     # 11
        'Cosmic'],      # 12
    'table': {
        0: 11,          # Other
        1: 0,           # CCQEL
        2: 6,           # NCQEL
        3: 1,           # CCNuPtoLPPiPlus
        4: 1,           # CCNuNtoLPPiZero
        5: 1,           # CCNuNtoLNPiPlus
        6: 7,           # NCNuPtoNuPPiZero
        7: 7,           # NCNuPtoNuNPiPlus
        8: 7,           # NCNuNtoNuNPiZero
        9: 7,           # NCNuNtoNuPPiMinus
        10: 1,          # CCNuBarNtoLNPiMinus
        11: 1,          # CCNuBarPtoLNPiZero
        12: 1,          # CCNuBarPtoLPPiMinus
        13: 7,          # NCNuBarPtoNuBarPPiZero
        14: 7,          # NCNuBarPtoNuBarNPiPlus
        15: 7,          # NCNuBarNtoNuBarNPiZero
        16: 7,          # NCNuBarNtoNuBarPPiMinus
        17: 5,          # CCOtherResonant
        18: 11,         # NCOtherResonant
        19: 4,          # CCMEC
        20: 10,         # NCMEC
        21: 11,         # IMD
        91: 2,          # CCDIS
        92: 8,          # NCDIS
        96: 9,          # NCCoh
        97: 3,          # CCCoh
        98: 11,         # ElasticScattering
        99: 11,         # InverseMuDecay
        100: 12         # CosmicMuon
    }
})

"""Map to all categories (Total = 19)"""
MAP_ALL_CAT = DotMap({
    'name': 't_all_cat',
    'categories': 24,
    'labels': [
        'Nuel-CC-QEL',  # 0
        'Nuel-CC-RES',  # 1
        'Nuel-CC-DIS',  # 2
        'Nuel-CC-COH',  # 3
        'Nuel-CC-MEC',  # 4
        'Nuel-CC-OTHER',  # 5
        'Nuel-NC-QEL',  # 6
        'Nuel-NC-RES',  # 7
        'Nuel-NC-DIS',  # 8
        'Nuel-NC-COH',  # 9
        'Nuel-NC-MEC',  # 10
        'Nuel-NC-OTHER',  # 11
        'Numu-CC-QEL',  # 12
        'Numu-CC-RES',  # 13
        'Numu-CC-DIS',  # 14
        'Numu-CC-COH',  # 15
        'Numu-CC-MEC',  # 16
        'Numu-CC-OTHER',  # 17
        'Numu-NC-QEL',  # 18
        'Numu-NC-RES',  # 19
        'Numu-NC-DIS',  # 20
        'Numu-NC-COH',  # 21
        'Numu-NC-MEC',  # 22
        'Numu-NC-OTHER',  # 23
        'Cosmic'],      # 24
    'table': {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (0, 4): 4,
        (0, 5): 5,
        (0, 6): 6,
        (0, 7): 7,
        (0, 8): 8,
        (0, 9): 9,
        (0, 10): 10,
        (0, 11): 11,
        (0, 12): 24,
        (1, 0): 12,
        (1, 1): 13,
        (1, 2): 14,
        (1, 3): 15,
        (1, 4): 16,
        (1, 5): 17,
        (1, 6): 18,
        (1, 7): 19,
        (1, 8): 20,
        (1, 9): 21,
        (1, 10): 22,
        (1, 11): 23,
        (1, 12): 24,
    }
})

"""Map a cosmic flag (Total = 2)"""
MAP_COSMIC_CAT = DotMap({
    'name': 't_cosmic_cat',
    'categories': 1,
    'labels': [
        'Cosmic',       # 0
        'Beam'],        # 1
    'table': {
        (0, 0): 0,
        (0, 1): 0,
        (0, 2): 0,
        (0, 3): 0,
        (0, 4): 0,
        (0, 5): 0,
        (0, 6): 0,
        (0, 7): 0,
        (0, 8): 0,
        (0, 9): 0,
        (0, 10): 0,
        (0, 11): 0,
        (0, 12): 1,
        (1, 0): 0,
        (1, 1): 0,
        (1, 2): 0,
        (1, 3): 0,
        (1, 4): 0,
        (1, 5): 0,
        (1, 6): 0,
        (1, 7): 0,
        (1, 8): 0,
        (1, 9): 0,
        (1, 10): 0,
        (1, 11): 0,
        (1, 12): 1,
    }
})

"""Map to full_combined categories (Total = 5)"""
MAP_FULL_COMB_CAT = DotMap({
    'name': 't_comb_cat',
    'categories': 3,
    'labels': [
        'Nuel-CC',      # 0
        'Numu-CC',      # 1
        'NC',           # 2
        'Cosmic'],      # 3
    'table': {
        (0, 0): 0,
        (0, 1): 0,
        (0, 2): 0,
        (0, 3): 0,
        (0, 4): 0,
        (0, 5): 0,
        (0, 6): 2,
        (0, 7): 2,
        (0, 8): 2,
        (0, 9): 2,
        (0, 10): 2,
        (0, 11): 2,
        (0, 12): 3,
        (1, 0): 1,
        (1, 1): 1,
        (1, 2): 1,
        (1, 3): 1,
        (1, 4): 1,
        (1, 5): 1,
        (1, 6): 2,
        (1, 7): 2,
        (1, 8): 2,
        (1, 9): 2,
        (1, 10): 2,
        (1, 11): 2,
        (1, 12): 3,
    }
})

"""Map to nc_nu_combined categories (Total = 14)"""
MAP_NU_NC_COMB_CAT = DotMap({
    'name': 't_nu_nc_cat',
    'categories': 18,
    'labels': [
        'Nuel-CC-QEL',  # 0
        'Nuel-CC-RES',  # 1
        'Nuel-CC-DIS',  # 2
        'Nuel-CC-COH',  # 3
        'Nuel-CC-MEC',  # 4
        'Nuel-CC-OTHER',  # 5
        'Numu-CC-QEL',  # 6
        'Numu-CC-RES',  # 7
        'Numu-CC-DIS',  # 8
        'Numu-CC-COH',  # 9
        'Numu-CC-MEC',  # 10
        'Numu-CC-OTHER',  # 11
        'NC-QEL',       # 12
        'NC-RES',       # 13
        'NC-DIS',       # 14
        'NC-COH',       # 15
        'NC-MEC',       # 16
        'NC-OTHER',     # 17
        'Cosmic'],      # 18
    'table': {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (0, 4): 4,
        (0, 5): 5,
        (0, 6): 12,
        (0, 7): 13,
        (0, 8): 14,
        (0, 9): 15,
        (0, 10): 16,
        (0, 11): 17,
        (0, 12): 18,
        (1, 0): 6,
        (1, 1): 7,
        (1, 2): 8,
        (1, 3): 9,
        (1, 4): 10,
        (1, 5): 11,
        (1, 6): 12,
        (1, 7): 13,
        (1, 8): 14,
        (1, 9): 15,
        (1, 10): 16,
        (1, 11): 17,
        (1, 12): 18,
    }
})

"""Map to nc_combined categories (Total = 11)"""
MAP_NC_COMB_CAT = DotMap({
    'name': 't_nc_cat',
    'categories': 13,
    'labels': [
        'Nuel-CC-QEL',  # 0
        'Nuel-CC-RES',  # 1
        'Nuel-CC-DIS',  # 2
        'Nuel-CC-COH',  # 3
        'Nuel-CC-MEC',  # 4
        'Nuel-CC-OTHER',  # 5
        'Numu-CC-QEL',  # 6
        'Numu-CC-RES',  # 7
        'Numu-CC-DIS',  # 8
        'Numu-CC-COH',  # 9
        'Numu-CC-MEC',  # 10
        'Numu-CC-OTHER',  # 11
        'NC',           # 12
        'Cosmic'],      # 13
    'table': {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (0, 4): 4,
        (0, 5): 5,
        (0, 6): 12,
        (0, 7): 12,
        (0, 8): 12,
        (0, 9): 12,
        (0, 10): 12,
        (0, 11): 12,
        (0, 12): 13,
        (1, 0): 6,
        (1, 1): 7,
        (1, 2): 8,
        (1, 3): 9,
        (1, 4): 10,
        (1, 5): 11,
        (1, 6): 12,
        (1, 7): 12,
        (1, 8): 12,
        (1, 9): 12,
        (1, 10): 12,
        (1, 11): 12,
        (1, 12): 13,
    }
})
