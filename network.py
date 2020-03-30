from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import GaussianNoise


def EITNN_Network(input_size, image_size, noise_level=0):
	model = Sequential()

	model.add(GaussianNoise(noise_level, input_shape=(input_size,)))

	model.add(Dense(4 * image_size * image_size))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Dense(image_size * image_size))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Dense(image_size * image_size))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Reshape((image_size, image_size, 1)))

	model.add(Conv2D(32, (3, 3), padding="same", use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), padding="same", use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Conv2DTranspose(1, (3, 3), padding="same", use_bias=False))
	model.add(Reshape((image_size * image_size,)))  # (image_size * image_size)

	return model

