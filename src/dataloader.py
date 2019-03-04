import numpy as np
import tensorflow as tf
import imageio


def to to_one_hot(array, num_classes=3):
	"""This function coverts the labels into a one hot vector.

	Keyword arguments:
	array -- The numpy array of labels(masks) of shape (number_samples, img_width, img_height).
	num_classes -- The number of classes (default 3)

	Returns:
	array_one_hot -- Numpy array of shape (number_samples, img_width, img_height, number_classes).
	"""

	sess = tf.Session(); 
	array_one_hot = []
	with sess.as_default():
	    for i in range(len(array)):
	        array_one_hot.append(tf.one_hot(array[i], num_classes).eval())
	return array_one_hot


def load_data(training_list, testing_list):
	"""This function loads the data required for the network.

	Keyword arguments:
	training_list -- List of training file names.
	testing_list -- List of testing file names. 

	Returns:
	X_train -- Numpy array of shape (number__train_samples, img_width, img_height, number_features).
	X_test -- Numpy array of shape (number_test_samples, img_width, img_height, number_features).
	y_train -- Numpy array of shape (number_training_samples, img_width, img_height, number_classes).
	"""


	X_train = []
	for index, name in enumerate(training_list):
		current = np.load('features/' + name + '.npy')
		if len(current.shape) < 3:
			current = np.expand_dims(current, axis=-1)
	    X_train.append(current)

	X_test = []
	for index, name in enumerate(testing_list):
		current = np.load('features/' + name + '.npy')
		if len(current.shape) < 3:
			current = np.expand_dims(current, axis=-1)
	    X_test.append(current)

	y_2d = []
	for index, name in enumerate(training_list):
		y_2d.append(imageio.imread('masks/' + name + '.png'))

	y_train = to_one_hot(y_2d)

	return X_train, X_test, y_train


