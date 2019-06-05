import talos as ta

import matplotlib
import matplotlib.pyplot as plt

import pandas

import astetik as ast

r = ta.Reporting('parameter_model_1.csv')

ast.box(r.data, "learning_rate", "val_mean_absolute_error", save=True, title='learning_rate_val')
ast.box(r.data, "batch_size", "val_mean_absolute_error", save=True, title='batch_size_val')
ast.box(r.data, "dense", "val_mean_absolute_error", save=True, title='dense_val')
ast.box(r.data, "dropout", "val_mean_absolute_error", save=True, title='dropout_val')

ast.box(r.data, "learning_rate", "mean_absolute_error", save=True, title='learning_rate')
ast.box(r.data, "batch_size", "mean_absolute_error", save=True, title='batch_size')
ast.box(r.data, "dense", "mean_absolute_error", save=True, title='dense')
ast.box(r.data, "dropout", "mean_absolute_error", save=True, title='dropout')
