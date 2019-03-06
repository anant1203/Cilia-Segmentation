from keras import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
import numpy as np
import imageio
import glob
import os
import argparse
import dataloader
import random
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import cv2


def get_data(X_train, y_train, num_train):
    """
    This function generates training samples.

    Parameters
    ----------
    X_train : List
        List of training samples features.
    y_train: Numpy array
        Target masks
    num_train: Int
        The number of training samples.

    Yields
    ----------
    sample_feature : numpy array
        The features of a particular sample.
    sample_label : numpy array
        The label (mask) of a particular sample.
    """

    while True:
        for i in range(num_train):
            sample_feature = np.array([X_train[i]])
            sample_label = np.array([y_train[i]])
            yield sample_feature, sample_label


def get_val_data(X_train, y_train, num_train, num_samples):
    """
    This function generates validation samples.

    Parameters
    ----------
    X_train : List
        List of training samples features.
    y_train: Numpy array
        Target masks
    num_train: Int
        The number of training samples.
    num_samples: Int
        The total number of samples.

    Yields
    ----------
    sample_feature : numpy array
        The features of a particular sample.
    sample_label : numpy array
        The label (mask) of a particular sample.
    """

    while True:
        for i in range(num_train, num_samples):
            sample_feature = np.array([X_train[i]])
            sample_label = np.array([y_train[i]])
            yield sample_feature, sample_label


def read_file(file):
    """
    This function reads a text file and returns
    the contents as a list (split linewise).

    Parameters
    ----------
    file : str
        Path to the text file.

    Returns
    ----------
    array : list
        List of file contents (split linewise).
    """

    with open(file, "r") as ins:
        array = []
        for line in ins:
            array.append(line.rstrip('\n'))
    return array


def down_block(inputs, num_filters, kernel_dims=3):
    """
    This function creates the contracting path blocks used in the UNET.

    Parameters
    ----------
    inputs : tensor
        Input tensor to this block.
    num_filters : int
        The number of filters to be used in this block's convolution layers.
    kernel_dims: int, optional
        The kernel dimension used in this block's convolution layers
        (the default is 3).

    Returns
    -------
    x : tensor
        The processed tensor after passing through this block's layers.
    """

    x = Conv2D(
        filters=num_filters,
        kernel_size=kernel_dims,
        padding='same',
        activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(
        filters=num_filters,
        kernel_size=kernel_dims,
        padding='same',
        activation='relu')(x)
    x = BatchNormalization()(x)
    return x


def up_block(
        c_large,
        c_small,
        num_filters,
        strides_dims=(
            2,
            2),
        kernel_dims=3):
    """
    This function creates the expansive path blocks used in the UNET.

    Parameters
    ----------
    c_large : tensor
        Output of the previous block.
    c_small : tensor
        Output of the corresponding contracting path block.
    num_filters: int
        The number of filters to be used in this block's layers.
    strides_dims : tuple
        The stride size to be used in this block's deconvolution layers
        (the default is (2, 2)).
    kernel_dims : int
        The kernel dimension used in this block's convolution layers
        (the default is 3).

    Returns
    -------
    u : tensor
        The processed tensor after passing through this block's layers.
    """

    u = Conv2DTranspose(
        num_filters,
        kernel_size=kernel_dims,
        strides=strides_dims,
        padding='same')(c_large)
    u = concatenate([u, c_small])
    u = down_block(u, num_filters)
    return u


def get_unet(inp, num_classes=3, num_down_blocks=5, num_up_blocks=4):
    """
    This function creates a UNET model used in image segmentation.
    Reference - https://arxiv.org/abs/1505.04597

    Parameters
    ----------
    inp : tensor
        Input tensor.
    num_classes : int, optional
        The number of classes (the default is 3).
    num_down_blocks: int, optional
        The number of contractive blocks in the Unet the default is 5).
    num_up_block : int, optional
        The number of expansive blocks in the Unet (the default is 4).

    Returns
    -------
    model : Model object
        The created Model object, as per specifictions.
    """

    # c -- List to store the outputs of each before pooling block.
    c = []

    # p -- List to store the outputs after pooling.
    p = [inp]

    # Number of filters start at 16. They keep doubling during contraction and
    # halving during expansion.

    num_filters = 16

    # Here we pass the data through the 5 contracting path block.
    for i in range(num_down_blocks):
        c_i = down_block(p[i], num_filters)
        num_filters *= 2
        p_i = MaxPooling2D((2, 2))(c_i)
        c.append(c_i)
        p.append(p_i)

    num_filters *= 1 / 2

    # Here we pass the data through the 4 expansive path block.
    for i in range(num_up_blocks):
        num_filters *= 1 / 2
        num_filters = int(num_filters)
        c_i = up_block(c[-1], c[-2 * (i + 1)], num_filters)
        c.append(c_i)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c[-1])

    model = Model(inputs=[inp], outputs=[outputs])

    return model


if __name__ == "__main__":
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(
        description=('Trains the model and outputs predictions.'),
        add_help='How to use', prog='model.py <args>')

    # Required arguments
    parser.add_argument("-d", "--data_path", required=True,
                        help=("Provide the path to the data folder"))

    # Optional arguments
    parser.add_argument("-e", "--num_epochs", default=100,
                        help=("Set the number of epochs"))

    parser.add_argument("-v", "--val_split", default=0.7,
                        help=("Set the training / validiation split ratio",
                              " .Set to 1 to use all data for training"))

    args = vars(parser.parse_args())

    # Getting the names of the training / testing files.
    training_files = read_file(args['data_path'] + '/train.txt')
    testing_files = read_file(args['data_path'] + '/test.txt')

    # Loading the data.
    X_train, X_test, y_train = dataloader.load_data(
        args['data_path'], training_files, testing_files)

    num_samples = len(X_train)

    # Forcing the training samples to be greater than 0.
    num_train = max(1, int(min(1, float(args['val_split']) * num_samples)))

    num_features = X_train[0].shape[2]

    # None is passed as the first two parameters
    # because the images have variable dimensions.
    inp = Input((None, None, num_features))
    model = get_unet(inp)

    model.compile(
        optimizer=Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"])

    # Training the model with or without validation data.
    if float(args['val_split']) >= 1:
        results = model.fit_generator(
            get_data(
                X_train, y_train, num_train), epochs=int(
                args['num_epochs']), steps_per_epoch=num_train)

    else:
        results = model.fit_generator(
            get_data(
                X_train, y_train, num_train), epochs=int(
                args['num_epochs']), steps_per_epoch=num_train, validation_data=get_val_data(
                X_train, y_train, num_train, num_samples), validation_steps=(
                    num_samples - num_train))

    num_test = len(X_test)

    predictions = []
    final_predictions = []

    for i in range(num_test):
        predictions.append(model.predict(np.array([X_test[i]]))[0])

    # Using argmax to decode the one hot encoding.
    for i in range(num_test):
        final_predictions.append(np.argmax(predictions[i], axis=2))

    if not os.path.exists(args['data_path'] + '/predictions'):
        os.mkdir(args['data_path'] + '/predictions')

    for i, name in enumerate(testing_files):
        file_name = name + '.png'
        arr = final_predictions[i]
        cv2.imwrite(args['data_path'] + '/predictions/' + file_name, arr)
