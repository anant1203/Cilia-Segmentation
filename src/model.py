from keras import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
import os
import random
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


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


def up_block(c_large, c_small, num_filters, strides_dims=(2, 2), kernel_dims=3):
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
    """This function creates a UNET model used in image segmentation.
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

    num_filters *= 1/2

    # Here we pass the data through the 4 expansive path block.
    for i in range(num_up_blocks):
        num_filters *= 1/2
        num_filters = int(num_filters)
        c_i = up_block(c[-1], c[-2*(i+1)], num_filters)
        c.append(c_i)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c[-1])

    model = Model(inputs=[inp], outputs=[outputs])
return model
