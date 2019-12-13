# TextDataHandler, handles the Tensorflow dataset built from the input .txt files
# Author: Josh Tingey
# Email: j.tingey.16@ucl.ac.uk

import os
import tensorflow as tf
import matplotlib.pyplot as plt

class DataHandler:
    def __init__(self, inputDir, config):
        self.file_names = os.listdir(inputDir)
        self.config = config
        self.dataset = self.load()

    def load(self):
        files_ds = tf.data.Dataset.from_tensor_slices(self.file_names)
        ds = files_ds.interleave(tf.data.TextLineDataset, cycle_length=len(self.file_names), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        parsed_ds = ds.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # Decode the text file format

        parsed_ds = parsed_ds.cache()  # Cache data into memory for use after first epoch
        parsed_ds = parsed_ds.batch(self.config.batch_size)
        parsed_ds = parsed_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return parsed_ds

    def parse(self, text):
        txt_array = tf.strings.split(text)  # Split line at " " into an array of strings

        label = tf.strings.to_number(txt_array[0], out_type=tf.int32)
        pars = tf.strings.to_number(txt_array[1:9], out_type=tf.float32)
        h_txt = tf.strings.to_number(txt_array[9:((self.config.img_size*self.config.img_size)+9)], out_type=tf.float32)
        t_txt = tf.strings.to_number(txt_array[((self.config.img_size*self.config.img_size)+9):], out_type=tf.float32)

        # Reshape and stack the two channels into a single tensor
        h_img = tf.reshape(h_txt, [self.config.img_size, self.config.img_size])
        t_img = tf.reshape(t_txt, [self.config.img_size, self.config.img_size])
        image = tf.stack([h_img, t_img])

        return image, label

    def plot(self, num):
        for plot, ev in enumerate(self.dataset.take(num)):
            fig = plt.figure()
            a = fig.add_subplot(1, 2, 1)
            imgplot = plt.imshow(ev[0][0][0])
            a.set_title('Hit Channel')
            plt.colorbar(orientation='horizontal')
            a = fig.add_subplot(1, 2, 2)
            imgplot = plt.imshow(ev[0][0][1])
            a.set_title('Time Channel')
            plt.colorbar(orientation='horizontal')
            plt.savefig("plot_{}.png".format(plot))