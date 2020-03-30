import tensorflow as tf
import keras.backend as K

from utils import load_dataset


def WPMSE(laplace=True):
	kernel = load_dataset("./spatial_sensitivity_matrix.csv")
	kernel = tf.convert_to_tensor(kernel, dtype=tf.float32)

	def loss(y_true, y_pred):
		if laplace:
			y_true = tf.matmul(y_true, kernel)
		return K.mean(K.square(y_true - y_pred))

	return loss

