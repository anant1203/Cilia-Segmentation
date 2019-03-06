import numpy as np
import tensorflow as tf
import imageio


def to_one_hot(array, num_classes=3):
    """
    This function coverts the labels into a one hot vector.

    Parameters
    ----------
    array : numpy array
        The numpy array of labels(masks) of
        shape (number_samples, img_width, img_height).
    num_classes : int, optional
        The number of classes (the default is 3)

    Returns
    -------
    array_one_hot : numpy array
        Numpy array of
        shape (number_samples, img_width, img_height, number_classes).
    """

    sess = tf.Session()
    array_one_hot = []
    with sess.as_default():
        for i in range(len(array)):
            array_one_hot.append(tf.one_hot(array[i], num_classes).eval())
    return array_one_hot


def load_data(data_path, training_list, testing_list, num_classes=3):
    """
    This function loads the data required for the network.
    
    Parameters
    ----------
    training_list : List
        List of training file names.
    testing_list : List
        List of testing file names.
    num_classes : int, optional
        The number of classes (the default is 3)

    Returns
    -------
    X_train : numpy array
        Training data of
        shape (number_train_samples, img_width, img_height, number_features).
    X_test : numpy array
        Testing data of
        shape (number_train_samples, img_width, img_height, number_features).
    y_train : numpy array
        Training labels of
        shape (number_training_samples, img_width, img_height, number_classes).
    """

    X_train = []
    for index, name in enumerate(training_list):
        current = np.load(data_path + '/features/' + name + '.npy')
        if len(current.shape) < 3:
            current = np.expand_dims(current, axis=-1)
        X_train.append(current)

    X_test = []
    for index, name in enumerate(testing_list):
        current = np.load(data_path + '/features/' + name + '.npy')
        if len(current.shape) < 3:
            current = np.expand_dims(current, axis=-1)
        X_test.append(current)

    y_2d = []
    for index, name in enumerate(training_list):
        y_2d.append(imageio.imread(data_path + '/masks/' + name + '.png'))

    y_train = to_one_hot(y_2d)

    return X_train, X_test, y_train
