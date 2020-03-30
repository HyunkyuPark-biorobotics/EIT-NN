import numpy as np


def load_dataset(path, transpose=False):
	dataset = np.loadtxt(path, delimiter=",")
	if transpose:
		dataset = np.transpose(dataset)
	return dataset


def save_history(path, history):
	np.savetxt(path, history, delimiter=",")

