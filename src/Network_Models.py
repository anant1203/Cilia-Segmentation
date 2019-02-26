from keras import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob


def unet():
    inputs = Input((256, 256, 1))

# down branch

    c11 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c12 = Conv2D(64, (3, 3), activation='relu', padding='same')(c11)
    p1 = MaxPool2D((2, 2))(c12)

    c21 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c22 = Conv2D(128, (3, 3), activation='relu', padding='same')(c21)
    p2 = MaxPool2D((2, 2))(c22)

    c31 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c32 = Conv2D(256, (3, 3), activation='relu', padding='same')(c31)
    p3 = MaxPool2D((2, 2))(c32)

    c41 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c42 = Conv2D(512, (3, 3), activation='relu', padding='same')(c41)
    p4 = MaxPool2D((2, 2))(c42)

    c51 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c52 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c51)

    # Upbranch

    u61 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(c52)
    u62 = concatenate([u61, c42])
    u63 = Conv2D(512, (3, 3), activation='relu', padding='same')(u62)
    u64 = Conv2D(512, (3, 3), activation='relu', padding='same')(u63)

    u71 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(u64)
    u72 = concatenate([u71, c32])
    u73 = Conv2D(256, (3, 3), activation='relu', padding='same')(u72)
    u74 = Conv2D(256, (3, 3), activation='relu', padding='same')(u73)

    u81 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(u74)
    u82 = concatenate([u81, c22])
    u83 = Conv2D(128, (3, 3), activation='relu', padding='same')(u82)
    u84 = Conv2D(128, (3, 3), activation='relu', padding='same')(u83)

    u91 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(u84)
    u92 = concatenate([u91, c12])
    u93 = Conv2D(64, (3, 3), activation='relu', padding='same')(u92)
    u94 = Conv2D(64, (3, 3), activation='relu', padding='same')(u93)

    outputs = Conv2D(3, (1, 1), activation='sigmoid')(u94)

    model = Model(inputs=inputs, outputs=outputs)
    return model
