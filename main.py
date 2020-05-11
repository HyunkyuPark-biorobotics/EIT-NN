import os
import argparse
from keras import optimizers
from keras.callbacks import EarlyStopping

from network import EITNN_Network
from wpmse import WPMSE
from utils import load_dataset, save_history


if __name__ == '__main__':
	parser = argparse.ArgumentParser("EIT-NN")
	parser.add_argument('--name', type=str, default='temp', help='experiment name')
	parser.add_argument('--epochs', type=int, default=20000, help='epochs')
	parser.add_argument('--batch_size', type=int, default=256, help='batch size')
	parser.add_argument('--learning_rate', type=float, default=0.00006, help='Adam learning rate')
	parser.add_argument('--input_size', type=int, default=256, help='input size')
	parser.add_argument('--image_size', type=int, default=24, help='image size')
	parser.add_argument('--noise_level', type=float, default=0, help='noise level')
	parser.add_argument('--no_laplace', action='store_true', help='do *not* use Laplace smoothing')
	args = parser.parse_args()

	# Load dataset
	x_train = load_dataset("./dataset/voltage_train.csv", transpose=True)  # (num_samples, input_size)
	x_valid = load_dataset("./dataset/voltage_valid.csv", transpose=True)  # (num_samples, input_size)
	y_train = load_dataset("./dataset/cond_train.csv", transpose=True)  # (num_samples, image_size * image_size)
	y_valid = load_dataset("./dataset/cond_valid.csv", transpose=True)  # (num_samples, image_size * image_size)

	# Train network
	model = EITNN_Network(args.input_size, args.image_size, noise_level=args.noise_level)
	loss = SSAMSE(laplace=not args.no_laplace)
	optimizer = optimizers.Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999)
	model.compile(loss=loss, optimizer=optimizer, metrics=['mse'])

	early_stopping = EarlyStopping(patience=100, restore_best_weights=True)
	hist = model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
	                 validation_data=(x_valid, y_valid), callbacks=[early_stopping])

	# Save network and results
	path = os.path.join("checkpoints", args.name)
	if not os.path.exists(path):
		os.mkdir(path)

	save_history(os.path.join(path, "train_loss.txt"), hist.history["loss"])
	save_history(os.path.join(path, "val_loss.txt"), hist.history["val_loss"])
	model.save(os.path.join(path, "model.h5"))

