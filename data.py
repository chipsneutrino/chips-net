"""Data methods and preprocessing functionality

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This script allows for the preprocessing of .root hit map files into
tfrecords files in a parallelisable fashion. It can also be imported
as a module to generate tensorflow datasets from the tfrecords files.
"""

import os
import argparse
from joblib import Parallel, delayed
import multiprocessing

import uproot
import numpy as np
import tensorflow as tf

import config


def parse(serialised_example):
    """Parses a single serialised example into tf tensors."""
    features = {
        'category': tf.io.FixedLenFeature([], tf.string),
        'parameters': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(serialised_example, features)

    image = tf.io.decode_raw(example['image'], tf.float32)
    image = tf.reshape(image, [64, 64, 3])
    pars = tf.io.decode_raw(example['parameters'], tf.float32)

    return image, pars[6]


def dataset(dirs, par=0):
    """Returns a dataset formed from all the files in the input directories."""
    files = []  # Add all files in dirs to a list
    for d in dirs:
        for file in os.listdir(d):
            files.append(os.path.join(d, file))

    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=len(files),
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds = ds.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(64, drop_remainder=True)

    return ds


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_examples(true, reco):
    """Generates a list of examples from the input .root map file."""
    # Get the numpy arrays from the .root map file
    categories = true.array("true_type_")
    parameters = np.stack((true.array("true_vtx_x"),
                           true.array("true_vtx_y"),
                           true.array("true_vtx_z"),
                           true.array("true_dir_costheta"),
                           true.array("true_dir_phi"),
                           true.array("true_nu_energy"),
                           true.array("true_lep_energy")), axis=1)
    images = np.stack((reco.array("filtered_charge_map_origin"),
                       reco.array("filtered_time_map_vtx"),
                       reco.array("filtered_hit_hough_map_vtx")), axis=3)

    # Generate examples using feature map
    examples = []
    for i in range(len(categories)):
        feature_dict = {
            'category': _int64_feature(categories[i]),
            'parameters': _float_feature(parameters[i]),
            'height': _int64_feature(images.shape[0]),
            'width': _int64_feature(images.shape[1]),
            'depth': _int64_feature(images.shape[2]),
            'image': _bytes_feature(images[i].tostring())
        }
        examples.append(tf.train.Example(features=tf.train.Features(
            feature=feature_dict)))

    return examples


def write_examples(name, examples):
    """Write a list of TF Examples to a tfrecords file."""
    with tf.io.TFRecordWriter(name) as writer:
        for example in examples:
            writer.write(example.SerializeToString())


def preprocess_file(conf, file, out_dir):
    """Preprocess a .root map file into train, val and test tfrecords files."""
    examples = []
    file_u = uproot.open(file)
    try:
        examples = get_examples(file_u["true"], file_u["reco"])
    except Exception as err:  # Catch when there is an uproot exception
        print("Error:", type(err), err)
        return

    # Split into training, validation and testing samples
    val_split = int((1.0-conf.test_split-conf.val_split) * len(examples))
    test_split = int((1.0-conf.test_split) * len(examples))
    train_examples = examples[:val_split]
    val_examples = examples[val_split:test_split]
    test_examples = examples[test_split:]

    name, ext = os.path.splitext(file)
    base = os.path.basename(name)
    write_examples(os.path.join(out_dir, "train/", base + "_train.tfrecords"),
                   train_examples)
    write_examples(os.path.join(out_dir, "val/", base + "_val.tfrecords"),
                   val_examples)
    write_examples(os.path.join(out_dir, "test/", base + "_test.tfrecords"),
                   test_examples)


def preprocess_dir(conf, in_dir, out_dir, single):
    """Preprocess all the files from the input directory into tfrecords."""
    if not single:  # File independence allows for parallelisation
        Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(
            preprocess_file)(conf, os.path.join(in_dir, f), out_dir)
                for f in os.listdir(in_dir))
    else:  # For debugging we keep the option to use a single process
        for f in os.listdir(in_dir):
            preprocess_file(conf, os.path.join(in_dir, f), out_dir)


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description='CHIPS CVN Data')
    parser.add_argument('-i', '--in_dir', help='path to input directory')
    parser.add_argument('-o', '--out_dir', help='path to output directory')
    parser.add_argument('-c', '--conf', help='Config .json file',
                        default='config/preprocessing.json')
    parser.add_argument('-s', '--single', help='Use a single process',
                        default=False)
    return parser.parse_args()


def main():
    """Main function called by script."""
    args = parse_args()
    conf = config.get_config(args.conf)
    preprocess_dir(conf, args.in_dir, args.out_dir, args.single)


if __name__ == '__main__':
    main()
