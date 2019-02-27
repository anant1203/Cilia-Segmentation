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

def down_block(inputs, num_filters, kernel_dims=3):
    x = Conv2D(filters=num_filters, kernel_size=kernel_dims, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(filters=num_filters, kernel_size=kernel_dims, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    return x

def up_block(c_large, c_small, num_filters, strides_dims=(2,2), kernel_dims = 3):
    u = Conv2DTranspose(num_filters, kernel_size=kernel_dims, strides=strides_dims, padding='same')(c_large)
    u = concatenate([u, c_small])
    u = down_block(u, num_filters)
    return u

def get_unet(inp, num_down_blocks=5, num_up_blocks=4):
    c = []
    p = [inp]
    num_filters = 16

    for i in range(num_down_blocks):
        c_i = down_block(p[i], num_filters)
        num_filters *= 2
        p_i = MaxPooling2D((2, 2))(c_i)
        c.append(c_i)
        p.append(p_i)
    num_filters *= 1/2
    for i in range(num_up_blocks):
        num_filters *= 1/2
        num_filters = int(num_filters)
        c_i = up_block(c[-1], c[-2*(i+1)], num_filters)
        c.append(c_i)

    outputs = Conv2D(1, (1, 1), activation='softmax')(c[-1])

    model = Model(inputs=[inp], outputs=[outputs])
    return model
