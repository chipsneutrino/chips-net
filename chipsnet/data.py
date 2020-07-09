# -*- coding: utf-8 -*-

"""Data creation and reading module.

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
from particle import Particle


class Reader:
    """Generates tf datasets for training/evaluation from the configuration.

    These can be read on the fly by Tensorflow so that the entire dataset does
    not need to be loaded into memory.
    """

    def __init__(self, config):
        """Initialise the Reader.

        Args:
            config (dotmap.DotMap): configuration namespace
        """
        self.config = config
        self.train_dirs = [
            os.path.join(in_dir, "train") for in_dir in config.data.input_dirs
        ]
        self.val_dirs = [
            os.path.join(in_dir, "val") for in_dir in config.data.input_dirs
        ]
        self.test_dirs = [
            os.path.join(in_dir, "test") for in_dir in config.data.input_dirs
        ]

        self.image_shape = [
            self.config.data.img_size[0],
            self.config.data.img_size[1],
            len(self.config.data.channels),
        ]

    @tf.function
    def parse(self, serialised_example):
        """Parse a single serialised example into both an input and labels dict.

        Args:
            serialised_example (tf.Example): a single example from .tfrecords file

        Returns:
            Tuple[dict, dict]: (inputs dict, labels dict)
        """
        features = {
            "inputs_image": tf.io.FixedLenFeature([], tf.string),
            "inputs_other": tf.io.FixedLenFeature([], tf.string),
            "labels_i": tf.io.FixedLenFeature([], tf.string),
            "labels_f": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(serialised_example, features)

        inputs, labels = {}, {}  # The two dictionaries to fill

        # Decode and reshape the 'image' into a tf tensor, reshape, then cast correctly and scale
        image = tf.io.decode_raw(example["inputs_image"], tf.uint8)
        image = tf.reshape(image, self.image_shape)
        image = tf.cast(image, tf.float32) / 256.0  # Cast to float and salce to [0,1]

        # Unstack the channels and reassemble if needed later
        unstacked = tf.unstack(image, axis=2)
        channels = []
        for i, enabled in enumerate(self.config.data.channels):
            if enabled:
                if self.config.data.augment:

                    # Apply the factor scaling of the bin contents
                    factor_shift = tf.random.normal(
                        shape=self.config.data.img_size,
                        mean=(1.0 + self.config.data.aug_factor_mean[i]),
                        stddev=self.config.data.aug_factor_sigma[i],
                        dtype=tf.float32,
                    )
                    unstacked[i] = tf.math.multiply(unstacked[i], factor_shift)

                    # Apply the absolute shifting of the bin contents
                    abs_shift = tf.random.normal(
                        shape=self.config.data.img_size,
                        mean=(self.config.data.aug_abs_mean[i]),
                        stddev=self.config.data.aug_abs_sigma[i],
                        dtype=tf.float32,
                    )
                    unstacked[i] = tf.math.add(unstacked[i], abs_shift)

                channels.append(unstacked[i])

        # Choose to either stack the channels back into a single tensor or keep them seperate
        if self.config.data.seperate_channels:
            for i, input_image in enumerate(channels):
                inputs["image_" + str(i)] = tf.expand_dims(input_image, 2)
        else:
            inputs["image_0"] = tf.stack(channels, axis=2)

        # Decode the other inputs and append to inputs dictionary
        inputs_other = tf.io.decode_raw(example["inputs_other"], tf.float32)
        inputs["r_total_digi_q"] = inputs_other[0]
        inputs["r_first_ring_height"] = inputs_other[1]
        inputs["r_vtx_x"] = inputs_other[2]
        inputs["r_vtx_y"] = inputs_other[3]
        inputs["r_vtx_z"] = inputs_other[4]
        inputs["r_vtx_t"] = inputs_other[5]
        inputs["r_dir_theta"] = inputs_other[6]
        inputs["r_dir_phi"] = inputs_other[7]

        # Decode integer labels and append to labels dictionary
        labels_i = tf.io.decode_raw(example["labels_i"], tf.int32)
        labels[MAP_NU_TYPE["name"]] = labels_i[0]
        labels[MAP_SIGN_TYPE["name"]] = labels_i[1]
        labels[MAP_INT_TYPE["name"]] = labels_i[2]
        labels[MAP_CC_CAT["name"]] = labels_i[3]
        labels[MAP_NC_CAT["name"]] = labels_i[4]
        labels[MAP_FINAL_CAT["name"]] = labels_i[5]
        labels[MAP_ALL_CAT["name"]] = labels_i[6]
        labels[MAP_COSMIC_CAT["name"]] = labels_i[7]
        labels[MAP_FULL_COMB_CAT["name"]] = labels_i[8]
        labels[MAP_NU_NC_COMB_CAT["name"]] = labels_i[9]
        labels[MAP_NC_COMB_CAT["name"]] = labels_i[10]
        labels["prim_total"] = labels_i[11]
        labels["prim_p"] = labels_i[12]
        labels["prim_cp"] = labels_i[13]
        labels["prim_np"] = labels_i[14]
        labels["prim_g"] = labels_i[15]

        # Decode float labels and append to the labels dictionary
        labels_f = tf.io.decode_raw(example["labels_f"], tf.float32)
        labels["t_vtx_x"] = labels_f[0]
        labels["t_vtx_y"] = labels_f[1]
        labels["t_vtx_z"] = labels_f[2]
        labels["t_vtx_t"] = labels_f[3]
        labels["t_nu_energy"] = labels_f[4]
        labels["t_lep_energy"] = labels_f[5]

        # Append labels to inputs if needed for multitask network
        if self.config.model.learn_weights:
            for label in self.config.model.labels:
                inputs["input_" + label] = labels[label]

        return inputs, labels

    @tf.function
    def strip(self, inputs, labels):
        """Strip all labels except those needed in training/validation.

        Args:
            tuple[dict, dict]: (inputs dict, labels dict)

        Returns:
            tuple[dict, dict]: (inputs dict, stripped labels dict)
        """
        labels = {k: labels[k] for k in self.config.model.labels}
        return inputs, labels

    def dataset(self, dirs, strip=True):
        """Return a dataset formed from all the files in the input directories.

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
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        ds = ds.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if strip:
            ds = ds.map(self.strip, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return ds

    def training_ds(self, num_events, batch_size=None, strip=True):
        """Return the training dataset.

        Returns:
            tf.dataset: training dataset
        """
        ds = self.dataset(self.train_dirs, strip=strip)
        ds = ds.take(num_events)
        if batch_size is not None:
            ds = ds.batch(batch_size, drop_remainder=True)
        return ds

    def validation_ds(self, num_events, batch_size=None, strip=True):
        """Return the validation dataset.

        Returns:
            tf.dataset: validation dataset
        """
        ds = self.dataset(self.val_dirs, strip=strip)
        ds = ds.take(num_events)
        if batch_size is not None:
            ds = ds.batch(batch_size, drop_remainder=True)
        return ds

    def testing_ds(self, num_events, batch_size=None, strip=False):
        """Return the testing dataset.

        Returns:
            tf.dataset: testing dataset
        """
        ds = self.dataset(self.test_dirs, strip=strip)
        ds = ds.take(num_events)
        if batch_size is not None:
            ds = ds.batch(batch_size, drop_remainder=True)
        return ds

    def training_df(self, num_events):
        """Return the training DataFrame.

        Args:
            num_events (int): number of events to include

        Returns:
            pd.DataFrame: training sample DataFrame
        """
        return df_from_ds(self.training_ds(num_events, 256))

    def validation_df(self, num_events):
        """Return the validation DataFrame.

        Args:
            num_events (int): number of events to include

        Returns:
            pd.DataFrame: validation sample DataFrame
        """
        return df_from_ds(self.validation_ds(num_events, 256))

    def testing_df(self, num_events):
        """Return the testing DataFrame.

        Args:
            num_events (int): number of events to include

        Returns:
            pd.DataFrame: testing sample DataFrame
        """
        return df_from_ds(self.testing_ds(num_events, 256))


class Creator:
    """Generate tfrecords files from ROOT map files."""

    def __init__(self, config):
        """Initialise the Creator.

        Args:
            config (dotmap.DotMap): configuration namespace
        """
        self.config = config
        os.makedirs(config.create.out_dir, exist_ok=True)
        os.makedirs(os.path.join(config.create.out_dir, "train/"), exist_ok=True)
        os.makedirs(os.path.join(config.create.out_dir, "val/"), exist_ok=True)
        os.makedirs(os.path.join(config.create.out_dir, "test/"), exist_ok=True)

    def count_primaries(self, pdgs, energies):
        """Count the number of Cherenkov threshold passing primaries for each type in the event.

        We count the number of particles for...
            - protons (above the cherenkov threshold)
            - charged pions (above the cherenkov threshold)
            - neutral pions (above a few pair productions in energy)
            - photons (above a few pair productions in energy)
            - total (total above the their threshold)
        in the event, either (0, 1, 2, n). Anything above 2 is classified into the 'n' category

        Args:
            pdgs (np.array): primary particle pdgs
            energies (np.array): primary particle energies

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

    def lepton_energies(self, pdgs, energies):
        """Get the leading lepton energies for the events.

        Args:
            pdgs (np.array): primary particle pdgs
            energies (np.array): primary particle energies

        Returns:
            np.array: array of leading lepton energies
        """
        energy_list = []
        for ev_pdgs, ev_energies in zip(pdgs, energies):  # loop through all events
            energy = 0.0
            for i, pdg in enumerate(ev_pdgs):
                if pdg in [11, 13] and ev_energies[i] > energy:
                    energy = ev_energies[i]
            energy_list.append(energy)

        return np.asarray(energy_list)

    def gen_examples(self, true, reco):
        """Generate a list of examples from the input .root map file.

        Args:
            true (uproot TTree): true TTree from input file
            reco (uproot TTree): reco TTree from input file

        Returns:
            List[tf.train.Example]: list of examples
        """
        # First setup the input image
        channels = [
            "r_charge_map_vtx",
            "r_time_map_vtx",
            "r_hough_map_vtx",
        ]
        if self.config.create.all_maps:
            channels.append("r_charge_map_origin")
            channels.append("r_time_map_origin")
            channels.append("r_charge_map_iso")
            channels.append("r_time_map_iso")
        channel_images = [reco.array(channel) for channel in channels]
        inputs_image = np.stack(channel_images, axis=3)

        # Next setup the other inputs, mainly reconstructed variables
        inputs_other = np.stack(
            (  # Reco Parameters (floats)
                reco.array("r_total_digi_q"),
                reco.array("r_first_ring_height"),
                reco.array("r_vtxX") / self.config.create.par_scale[0],
                reco.array("r_vtxY") / self.config.create.par_scale[1],
                reco.array("r_vtxZ") / self.config.create.par_scale[2],
                reco.array("r_vtxT"),
                reco.array("r_dirTheta") / self.config.create.par_scale[3],
                reco.array("r_dirPhi") / self.config.create.par_scale[4],
            ),
            axis=1,
        )

        # Next setup the integer labels, we need to map to the different categories etc...
        n_arr = np.vectorize(MAP_NU_TYPE["table"].get)(true.array("t_nu"))
        s_arr = np.vectorize(MAP_SIGN_TYPE["table"].get)(true.array("t_nu"))
        i_arr = np.vectorize(MAP_INT_TYPE["table"].get)(true.array("t_code"))
        cc_arr = np.array([MAP_CC_CAT["table"][(n, i)] for n, i in zip(n_arr, i_arr)])
        nc_arr = np.array([MAP_NC_CAT["table"][(n, i)] for n, i in zip(n_arr, i_arr)])
        final_arr = np.array(
            [MAP_FINAL_CAT["table"][(n, i)] for n, i in zip(n_arr, i_arr)]
        )
        cat_arr = np.array([MAP_ALL_CAT["table"][(n, i)] for n, i in zip(n_arr, i_arr)])
        cosmic_arr = np.array(
            [MAP_COSMIC_CAT["table"][(n, i)] for n, i in zip(n_arr, i_arr)]
        )
        comb_arr = np.array(
            [MAP_FULL_COMB_CAT["table"][(n, i)] for n, i in zip(n_arr, i_arr)]
        )
        nu_nc_comb_arr = np.array(
            [MAP_NU_NC_COMB_CAT["table"][(n, i)] for n, i in zip(n_arr, i_arr)]
        )
        nc_comb_arr = np.array(
            [MAP_NC_COMB_CAT["table"][(n, i)] for n, i in zip(n_arr, i_arr)]
        )

        counts = self.count_primaries(
            true.array("t_p_pdgs"), true.array("t_p_energies")
        )

        labels_i = np.stack(
            (  # True Parameters (integers)
                n_arr,
                s_arr,
                i_arr,
                cc_arr,
                nc_arr,
                final_arr,
                cat_arr,
                cosmic_arr,
                comb_arr,
                nu_nc_comb_arr,
                nc_comb_arr,
                counts[:, 0],
                counts[:, 1],
                counts[:, 2],
                counts[:, 3],
                counts[:, 4],
            ),
            axis=1,
        ).astype(np.int32)

        lepton_energies = self.lepton_energies(
            true.array("t_p_pdgs"), true.array("t_p_energies")
        )

        labels_f = np.stack(
            (  # True Parameters (floats)
                true.array("t_vtxX"),
                true.array("t_vtxY"),
                true.array("t_vtxZ"),
                true.array("t_vtxT"),
                true.array("t_nuEnergy"),
                lepton_energies,
            ),
            axis=1,
        ).astype(np.float32)

        # print(labels_f)
        # print(labels_f.shape)
        # print(labels_f.dtype)

        examples = []  # Generate examples using a feature dict
        for i in range(len(labels_i)):
            feature_dict = {
                "inputs_image": bytes_feature(inputs_image[i].tostring()),
                "inputs_other": bytes_feature(inputs_other[i].tostring()),
                "labels_i": bytes_feature(labels_i[i].tostring()),
                "labels_f": bytes_feature(labels_f[i].tostring()),
            }
            examples.append(
                tf.train.Example(features=tf.train.Features(feature=feature_dict))
            )

        return examples

    def preprocess_files(self, num, files):
        """Preprocess joined .root map files into train, val and test tfrecords files.

        Args:
            num (int): job number
            files (list[str]): list of input files to use
        """
        examples = []
        print("job {}...".format(num))
        for file in files:
            file_u = uproot.open(file)
            examples.extend(self.gen_examples(file_u["true"], file_u["reco"]))

        # Split into training, validation and testing samples
        random.shuffle(examples)  # Shuffle the examples list
        val_split = int(
            (1.0 - self.config.create.split - self.config.create.split) * len(examples)
        )
        test_split = int((1.0 - self.config.create.split) * len(examples))
        train_examples = examples[:val_split]
        val_examples = examples[val_split:test_split]
        test_examples = examples[test_split:]

        write_examples(
            os.path.join(
                self.config.create.out_dir, "train/", str(num) + "_train.tfrecords"
            ),
            train_examples,
        )
        write_examples(
            os.path.join(
                self.config.create.out_dir, "val/", str(num) + "_val.tfrecords"
            ),
            val_examples,
        )
        write_examples(
            os.path.join(
                self.config.create.out_dir, "test/", str(num) + "_test.tfrecords"
            ),
            test_examples,
        )

    def run(self):
        """Preprocess all the files from the input directory into tfrecords."""
        files = []
        for directory in self.config.create.input_dirs:
            files.extend(
                [os.path.join(directory, file) for file in os.listdir(directory)]
            )

        random.seed(8)
        random.shuffle(files)  # Shuffle the file list

        file_lists = [
            files[n : n + self.config.create.join]
            for n in range(0, len(files), self.config.create.join)
        ]
        if self.config.create.parallel:  # File independence allows for parallelisation
            Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
                delayed(self.preprocess_files)(counter, f_list)
                for counter, f_list in enumerate(file_lists)
            )
        else:
            for counter, f_list in enumerate(file_lists):
                self.preprocess_files(counter, f_list)


"""Declare constants for use in primary particle counting"""
INDEX = 1.344  # for 405nm at ~4 degrees celcius
PROTON_THRESHOLD = math.sqrt(
    math.pow(Particle.from_pdgid(2212).mass, 2) / (1 - (1 / math.pow(INDEX, 2)))
)
CP_THRESHOLD = math.sqrt(
    math.pow(Particle.from_pdgid(211).mass, 2) / (1 - (1 / math.pow(INDEX, 2)))
)
NP_THRESHOLD = 20 * Particle.from_pdgid(11).mass
GAMMA_THRESHOLD = 20 * Particle.from_pdgid(11).mass


def df_from_ds(ds):
    """Create a pandas dataframe from a tf dataset.

    Args:
        ds (tf.dataset): input dataset
        num_events (int): number of events to include

    Returns:
        pd.DataFrame: dataFrame generated from the dataset
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


def bytes_feature(value):
    """Return a BytesList feature from a string/byte.

    Args:
        value (str): raw string format of an array

    Returns:
        tf.train.Feature: a BytesList feature
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_examples(name, examples):
    """Write a list of examples to a tfrecords file.

    Args:
        name (str): uutput .tfrecords file path
        examples (list[tf.train.Example]): list of examples
    """
    with tf.io.TFRecordWriter(name) as writer:
        for example in examples:
            writer.write(example.SerializeToString())


def get_map(name):
    """Get category mapping dict from name.

    Args:
        name (str): name of mapping

    Returns:
        dict: mapping dictionary
    """
    for cat_map in [
        MAP_NU_TYPE,
        MAP_INT_TYPE,
        MAP_CC_CAT,
        MAP_NC_CAT,
        MAP_FINAL_CAT,
        MAP_ALL_CAT,
        MAP_COSMIC_CAT,
        MAP_FULL_COMB_CAT,
        MAP_NU_NC_COMB_CAT,
        MAP_NC_COMB_CAT,
    ]:
        if cat_map["name"] == name:
            return cat_map
    return None


def binary_crossentropy(y_true, y_pred):
    """Return a standard binary crossentropy loss.

    Args:
        y_true (tf.tensor): true value
        y_pred (tf.tensor): predicted value

    Returns:
        tf.keras.loss: binary cross entropy function
    """
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)


"""Map to electron or muon types (Total = 2) (cosmic muons are included in this)"""
MAP_NU_TYPE = {
    "name": "t_nu_type",
    "categories": 1,
    "loss": binary_crossentropy,
    "labels": [r"$\nu_{e}$", r"$\nu_{\mu}$"],  # 0  # 1
    "table": {
        11: 0,  # el-
        -11: 0,  # el+
        12: 0,  # el neutrino
        -12: 0,  # el anti neutrino
        13: 1,  # mu-
        -13: 1,  # mu+
        14: 1,  # mu neutrino
        -14: 1,
    },  # mu anti neutrino
}

"""Map to particle vs anti-particle (Total = 2) (cosmic muons are included in this)"""
MAP_SIGN_TYPE = {
    "name": "t_sign_type",
    "categories": 1,
    "loss": binary_crossentropy,
    "labels": [r"$\nu$", r"$\bar{\nu}$"],  # 0  # 1
    "table": {
        11: 0,  # el-
        -11: 1,  # el+
        12: 0,  # el neutrino
        -12: 1,  # el anti neutrino
        13: 0,  # mu-
        -13: 1,  # mu+
        14: 0,  # mu neutrino
        -14: 1,
    },  # mu anti neutrino
}


def int_type_loss(y_true, y_pred):
    """Return a masked sparse categorical crossentropy loss.

    Args:
        y_true (tf.tensor): true value
        y_pred (tf.tensor): predicted value

    Returns:
        tf.keras.loss: sparse categorical crossentropy function
    """
    mask = tf.cast(tf.math.not_equal(y_true, 12), tf.float32)
    return tf.keras.losses.sparse_categorical_crossentropy(y_true * mask, y_pred * mask)


"""Map interaction types (Total = 13)
We put IMD, ElasticScattering and InverseMuDecay into 'NC-OTHER' for simplicity"""
MAP_INT_TYPE = {
    "name": "t_int_type",
    "categories": 12,
    "loss": int_type_loss,
    "labels": [
        "CC-QEL",  # 0
        "CC-RES",  # 1
        "CC-DIS",  # 2
        "CC-COH",  # 3
        "CC-MEC",  # 4
        "CC-OTHER",  # 5
        "NC-QEL",  # 6
        "NC-RES",  # 7
        "NC-DIS",  # 8
        "NC-COH",  # 9
        "NC-MEC",  # 10
        "NC-OTHER",  # 11
        "Cosmic",  # 12
    ],
    "table": {
        0: 11,  # Other
        1: 0,  # CCQEL
        2: 6,  # NCQEL
        3: 1,  # CCNuPtoLPPiPlus
        4: 1,  # CCNuNtoLPPiZero
        5: 1,  # CCNuNtoLNPiPlus
        6: 7,  # NCNuPtoNuPPiZero
        7: 7,  # NCNuPtoNuNPiPlus
        8: 7,  # NCNuNtoNuNPiZero
        9: 7,  # NCNuNtoNuPPiMinus
        10: 1,  # CCNuBarNtoLNPiMinus
        11: 1,  # CCNuBarPtoLNPiZero
        12: 1,  # CCNuBarPtoLPPiMinus
        13: 7,  # NCNuBarPtoNuBarPPiZero
        14: 7,  # NCNuBarPtoNuBarNPiPlus
        15: 7,  # NCNuBarNtoNuBarNPiZero
        16: 7,  # NCNuBarNtoNuBarPPiMinus
        17: 5,  # CCOtherResonant
        18: 11,  # NCOtherResonant
        19: 4,  # CCMEC
        20: 10,  # NCMEC
        21: 11,  # IMD
        91: 2,  # CCDIS
        92: 8,  # NCDIS
        96: 9,  # NCCoh
        97: 3,  # CCCoh
        98: 11,  # ElasticScattering
        99: 11,  # InverseMuDecay
        100: 12,  # CosmicMuon
    },
}


def cc_cat_loss(y_true, y_pred):
    """Return a masked sparse categorical crossentropy loss.

    Args:
        y_true (tf.tensor): true value
        y_pred (tf.tensor): predicted value

    Returns:
        tf.keras.loss: sparse categorical crossentropy function
    """
    mask = tf.cast(tf.math.not_equal(y_true, 5), tf.float32)
    return tf.keras.losses.sparse_categorical_crossentropy(y_true * mask, y_pred * mask)


"""Map cc categories (Total = 5)"""
MAP_CC_CAT = {
    "name": "t_cc_type",
    "categories": 5,
    "loss": cc_cat_loss,
    "labels": [
        "CC-QEL",  # 0
        "CC-RES",  # 1
        "CC-DIS",  # 2
        "CC-COH",  # 3
        "CC-OTHER",  # 4
        "Cosmic-NC",  # 5
    ],
    "table": {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (0, 4): 0,
        (0, 5): 4,
        (0, 6): 5,
        (0, 7): 5,
        (0, 8): 5,
        (0, 9): 5,
        (0, 10): 5,
        (0, 11): 5,
        (0, 12): 5,
        (1, 0): 0,
        (1, 1): 1,
        (1, 2): 2,
        (1, 3): 3,
        (1, 4): 0,
        (1, 5): 4,
        (1, 6): 5,
        (1, 7): 5,
        (1, 8): 5,
        (1, 9): 5,
        (1, 10): 5,
        (1, 11): 5,
        (1, 12): 5,
    },
}


def nc_cat_loss(y_true, y_pred):
    """Return a masked sparse categorical crossentropy loss.

    Args:
        y_true (tf.tensor): true value
        y_pred (tf.tensor): predicted value

    Returns:
        tf.keras.loss: sparse categorical crossentropy function
    """
    mask = tf.cast(tf.math.not_equal(y_true, 5), tf.float32)
    return tf.keras.losses.sparse_categorical_crossentropy(y_true * mask, y_pred * mask)


"""Map nc categories (Total = 5)"""
MAP_NC_CAT = {
    "name": "t_nc_type",
    "categories": 5,
    "loss": nc_cat_loss,
    "labels": [
        "NC-QEL",  # 0
        "NC-RES",  # 1
        "NC-DIS",  # 2
        "NC-COH",  # 3
        "NC-OTHER",  # 4
        "Cosmic-CC",  # 5
    ],
    "table": {
        (0, 0): 5,
        (0, 1): 5,
        (0, 2): 5,
        (0, 3): 5,
        (0, 4): 5,
        (0, 5): 5,
        (0, 6): 0,
        (0, 7): 1,
        (0, 8): 2,
        (0, 9): 3,
        (0, 10): 0,
        (0, 11): 4,
        (0, 12): 5,
        (1, 0): 5,
        (1, 1): 5,
        (1, 2): 5,
        (1, 3): 5,
        (1, 4): 5,
        (1, 5): 5,
        (1, 6): 0,
        (1, 7): 1,
        (1, 8): 2,
        (1, 9): 3,
        (1, 10): 0,
        (1, 11): 4,
        (1, 12): 5,
    },
}


def final_cat_loss(y_true, y_pred):
    """Return a masked sparse categorical crossentropy loss.

    Args:
        y_true (tf.tensor): true value
        y_pred (tf.tensor): predicted value

    Returns:
        tf.keras.loss: sparse categorical crossentropy function
    """
    mask = tf.cast(tf.math.not_equal(y_true, 15), tf.float32)
    return tf.keras.losses.sparse_categorical_crossentropy(y_true * mask, y_pred * mask)


"""Map final categories (Total = 15)"""
MAP_FINAL_CAT = {
    "name": "t_final_cat",
    "categories": 15,
    "loss": final_cat_loss,
    "labels": [
        r"$\nu_{e}$ CC-QEL",  # 0
        r"$\nu_{e}$ CC-RES",  # 1
        r"$\nu_{e}$ CC-DIS",  # 2
        r"$\nu_{e}$ CC-COH",  # 3
        r"$\nu_{e}$ CC-OTHER",  # 4
        r"$\nu_{\mu}$ CC-QEL",  # 5
        r"$\nu_{\mu}$ CC-RES",  # 6
        r"$\nu_{\mu}$ CC-DIS",  # 7
        r"$\nu_{\mu}$ CC-COH",  # 8
        r"$\nu_{\mu}$ CC-OTHER",  # 9
        "NC-QEL",  # 10
        "NC-RES",  # 11
        "NC-DIS",  # 12
        "NC-COH",  # 13
        "NC-OTHER",  # 14
        "Cosmic",  # 15
    ],
    "table": {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (0, 4): 0,
        (0, 5): 4,
        (0, 6): 10,
        (0, 7): 11,
        (0, 8): 12,
        (0, 9): 13,
        (0, 10): 10,
        (0, 11): 14,
        (0, 12): 15,
        (1, 0): 5,
        (1, 1): 6,
        (1, 2): 7,
        (1, 3): 8,
        (1, 4): 5,
        (1, 5): 9,
        (1, 6): 10,
        (1, 7): 11,
        (1, 8): 12,
        (1, 9): 13,
        (1, 10): 10,
        (1, 11): 14,
        (1, 12): 15,
    },
}


def all_cat_loss(y_true, y_pred):
    """Return a masked sparse categorical crossentropy loss.

    Args:
        y_true (tf.tensor): true value
        y_pred (tf.tensor): predicted value

    Returns:
        tf.keras.loss: sparse categorical crossentropy function
    """
    mask = tf.cast(tf.math.not_equal(y_true, 24), tf.float32)
    return tf.keras.losses.sparse_categorical_crossentropy(y_true * mask, y_pred * mask)


"""Map to all categories (Total = 19)"""
MAP_ALL_CAT = {
    "name": "t_all_cat",
    "categories": 24,
    "loss": all_cat_loss,
    "labels": [
        r"$\nu_{e}$ CC-QEL",  # 0
        r"$\nu_{e}$ CC-RES",  # 1
        r"$\nu_{e}$ CC-DIS",  # 2
        r"$\nu_{e}$ CC-COH",  # 3
        r"$\nu_{e}$ CC-MEC",  # 4
        r"$\nu_{e}$ CC-OTHER",  # 5
        r"$\nu_{e}$ NC-QEL",  # 6
        r"$\nu_{e}$ NC-RES",  # 7
        r"$\nu_{e}$ NC-DIS",  # 8
        r"$\nu_{e}$ NC-COH",  # 9
        r"$\nu_{e}$ NC-MEC",  # 10
        r"$\nu_{e}$ NC-OTHER",  # 11
        r"$\nu_{\mu}$ CC-QEL",  # 12
        r"$\nu_{\mu}$ CC-RES",  # 13
        r"$\nu_{\mu}$ CC-DIS",  # 14
        r"$\nu_{\mu}$ CC-COH",  # 15
        r"$\nu_{\mu}$ CC-MEC",  # 16
        r"$\nu_{\mu}$ CC-OTHER",  # 17
        r"$\nu_{\mu}$ NC-QEL",  # 18
        r"$\nu_{\mu}$ NC-RES",  # 19
        r"$\nu_{\mu}$ NC-DIS",  # 20
        r"$\nu_{\mu}$ NC-COH",  # 21
        r"$\nu_{\mu}$ NC-MEC",  # 22
        r"$\nu_{\mu}$ NC-OTHER",  # 23
        "Cosmic",  # 24
    ],
    "table": {
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
    },
}


"""Map a cosmic flag (Total = 2)"""
MAP_COSMIC_CAT = {
    "name": "t_cosmic_cat",
    "categories": 1,
    "loss": binary_crossentropy,
    "labels": ["Cosmic", "Beam"],  # 0  # 1
    "table": {
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
    },
}


def full_comb_loss(y_true, y_pred):
    """Return a masked sparse categorical crossentropy loss.

    Args:
        y_true (tf.tensor): true value
        y_pred (tf.tensor): predicted value

    Returns:
        tf.keras.loss: sparse categorical crossentropy function
    """
    mask = tf.cast(tf.math.not_equal(y_true, 3), tf.float32)
    return tf.keras.losses.sparse_categorical_crossentropy(y_true * mask, y_pred * mask)


"""Map to full_combined categories (Total = 5)"""
MAP_FULL_COMB_CAT = {
    "name": "t_comb_cat",
    "categories": 3,
    "loss": full_comb_loss,
    "labels": [r"$\nu_{e}$ CC", r"$\nu_{\mu}$ CC", "NC", "Cosmic"],  # 0  # 1  # 2  # 3
    "table": {
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
    },
}


def nu_nc_comb_loss(y_true, y_pred):
    """Return a masked sparse categorical crossentropy loss.

    Args:
        y_true (tf.tensor): true value
        y_pred (tf.tensor): predicted value

    Returns:
        tf.keras.loss: sparse categorical crossentropy function
    """
    mask = tf.cast(tf.math.not_equal(y_true, 18), tf.float32)
    return tf.keras.losses.sparse_categorical_crossentropy(y_true * mask, y_pred * mask)


"""Map to nc_nu_combined categories (Total = 14)"""
MAP_NU_NC_COMB_CAT = {
    "name": "t_nu_nc_cat",
    "categories": 18,
    "loss": nu_nc_comb_loss,
    "labels": [
        r"$\nu_{e}$ CC-QEL",  # 0
        r"$\nu_{e}$ CC-RES",  # 1
        r"$\nu_{e}$ CC-DIS",  # 2
        r"$\nu_{e}$ CC-COH",  # 3
        r"$\nu_{e}$ CC-MEC",  # 4
        r"$\nu_{e}$ CC-OTHER",  # 5
        r"$\nu_{\mu}$ CC-QEL",  # 6
        r"$\nu_{\mu}$ CC-RES",  # 7
        r"$\nu_{\mu}$ CC-DIS",  # 8
        r"$\nu_{\mu}$ CC-COH",  # 9
        r"$\nu_{\mu}$ CC-MEC",  # 10
        r"$\nu_{\mu}$ CC-OTHER",  # 11
        "NC-QEL",  # 12
        "NC-RES",  # 13
        "NC-DIS",  # 14
        "NC-COH",  # 15
        "NC-MEC",  # 16
        "NC-OTHER",  # 17
        "Cosmic",  # 18
    ],
    "table": {
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
    },
}


def nc_comb_loss(y_true, y_pred):
    """Return a masked sparse categorical crossentropy loss.

    Args:
        y_true (tf.tensor): true value
        y_pred (tf.tensor): predicted value

    Returns:
        tf.keras.loss: sparse categorical crossentropy function
    """
    mask = tf.cast(tf.math.not_equal(y_true, 13), tf.float32)
    return tf.keras.losses.sparse_categorical_crossentropy(y_true * mask, y_pred * mask)


"""Map to nc_combined categories (Total = 11)"""
MAP_NC_COMB_CAT = {
    "name": "t_nc_cat",
    "categories": 13,
    "loss": nc_comb_loss,
    "labels": [
        r"$\nu_{e}$ CC-QEL",  # 0
        r"$\nu_{e}$ CC-RES",  # 1
        r"$\nu_{e}$ CC-DIS",  # 2
        r"$\nu_{e}$ CC-COH",  # 3
        r"$\nu_{e}$ CC-MEC",  # 4
        r"$\nu_{e}$ CC-OTHER",  # 5
        r"$\nu_{mu}$ CC-QEL",  # 6
        r"$\nu_{mu}$ CC-RES",  # 7
        r"$\nu_{mu}$ CC-DIS",  # 8
        r"$\nu_{mu}$ CC-COH",  # 9
        r"$\nu_{mu}$ CC-MEC",  # 10
        r"$\nu_{mu}$ CC-OTHER",  # 11
        "NC",  # 12
        "Cosmic",  # 13
    ],
    "table": {
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
    },
}
