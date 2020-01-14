# Utilities
# Author: Josh Tingey
# Email: j.tingey.16@ucl.ac.uk

import matplotlib.pyplot as plt
import os


def plot_event(raw_images, vtx_images, index):
    fig = plt.figure()
    a = fig.add_subplot(1, 5, 1)
    plt.imshow(raw_images[index, :, :, 0])
    a.set_title('Hit Raw')
    plt.colorbar(orientation='horizontal')
    a = fig.add_subplot(1, 5, 2)
    plt.imshow(raw_images[index, :, :, 1])
    a.set_title('Hit Raw')
    plt.colorbar(orientation='horizontal')
    a = fig.add_subplot(1, 5, 3)
    plt.imshow(vtx_images[index, :, :, 0])
    a.set_title('Hit Vtx')
    plt.colorbar(orientation='horizontal')
    a = fig.add_subplot(1, 5, 4)
    plt.imshow(vtx_images[index, :, :, 1])
    a.set_title('Time Vtx')
    plt.colorbar(orientation='horizontal')
    a = fig.add_subplot(1, 5, 5)
    plt.imshow(vtx_images[index, :, :, 2])
    a.set_title('Hough Vtx')
    plt.colorbar(orientation='horizontal')
    plt.savefig("plot_{}.png".format(index))


# Plotting: Plot the history of a category based training
def plot_category_history(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# Plotting: Plot the history of a regression based training
def plot_regression_history(history):
    # summarize history for accuracy
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model error')
    plt.ylabel('mean absolute error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model square error')
    plt.ylabel('mean square error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# Plotting: Plot the (true-estimated) distribution
def plot_true_minus_estimation(diff, bins=50, hist_range=[-1500, 1500]):
    plt.hist(diff, bins=bins, range=hist_range)
    plt.ylabel('Events')
    plt.xlabel('True - Estimation')
    plt.show()


# Output: Save the history of a category based training
def save_category_history(train_history, output_file):
    print("utils: save_category_history()")

    name, ext = os.path.splitext(output_file)
    output = open(name + "_history.txt", "w")

    acc = train_history.history['acc']
    val_acc = train_history.history['val_acc']
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']

    # [acc, val_acc, loss, val_loss]
    for epoch in range(len(acc)):
        out = str(acc[epoch])+" "+str(val_acc[epoch])+" " + \
            str(loss[epoch])+" "+str(val_loss[epoch])+"\n"
        output.write(out)
    output.close()


# Output: Save the test output from a category based model
def save_category_output(categories, test_labels, test_parameters,
                         test_output, output_file):
    print("utils: save_category_output()")

    if len(test_labels) != len(test_parameters):
        print("Error: Arrays are not the same size")

    # [label, parameters, testOutput[Outputs for the different categories]]
    output = open(output_file, "w")
    for num in range(len(test_labels)):
        out = str(test_labels[num])
        out += (" " + str(test_parameters[num][0]) + " " +
                str(test_parameters[num][1]) + " " +
                str(test_parameters[num][2]) + " " +
                str(test_parameters[num][3]))
        out += (" " + str(test_parameters[num][4]) + " " +
                str(test_parameters[num][5]) + " " +
                str(test_parameters[num][6]) + " " +
                str(test_parameters[num][7]))
        for category in range(categories):
            out += (" " + str(test_output[num][category]))
        out += "\n"
        output.write(out)
    output.close()


# Output: Save the history of a regression based training
def save_regression_history(train_history, output_file):
    print("Output: save_regression_history")

    name, ext = os.path.splitext(output_file)
    output = open(name + "_history.txt", "w")

    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    mean_abs_err = train_history.history['mean_absolute_error']
    val_mean_abs_err = train_history.history['val_mean_absolute_error']
    mean_squared_err = train_history.history['mean_squared_error']
    val_mean_squared_err = train_history.history['val_mean_squared_error']

    for epoch in range(len(mean_abs_err)):
        out = str(loss[epoch]) + " " + str(val_loss[epoch]) + " "
        + str(mean_abs_err[epoch]) + " " + str(val_mean_abs_err[epoch]) + " "
        + str(mean_squared_err[epoch]) + " " + str(val_mean_squared_err[epoch])
        + "\n"
        output.write(out)
    output.close()


# Output: Save the test output from a regression based model
def save_regression_output(test_labels, test_parameters,
                           test_output, output_file):
    print("Output: save_regression_output")

    if len(test_labels) != len(test_output):
        print("Error: test_labels and test_output not the same length")

    # [label, parameters, test_output]
    output = open(output_file, "w")
    for num in range(len(test_labels)):
        out = str(test_labels[num])
        out += (" " + str(test_parameters[num][0]) + " " +
                str(test_parameters[num][1]) + " " +
                str(test_parameters[num][2]) + " " +
                str(test_parameters[num][3]))
        out += (" " + str(test_parameters[num][4]) + " " +
                str(test_parameters[num][5]) + " " +
                str(test_parameters[num][6]) + " " +
                str(test_parameters[num][7]))
        for par in range(len(test_output[num])):
            out += (" " + str(test_output[num][par]))
        out += "\n"
        output.write(out)
    output.close()
