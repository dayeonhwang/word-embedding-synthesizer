import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Implementation of a feedforward neural network(MLP) to use it as an autoencoder for word embeddings
# Requirements: TensorFlow, Pandas, Sklearn, Numpy

# load data (two word embeddings)
govt = pd.read_csv('govt_40.csv', encoding='ISO-8859-1') #(19928, 101)
books = pd.read_csv('books_40.csv', encoding='ISO-8859-1') #(17506,101)

# set random seed value
RANDOM_SEED = 101
tf.set_random_seed(RANDOM_SEED)

def init_weights(shape):
	"""weight initialization
	Note: use random value from normal distribution
	"""
	weights = tf.random_normal(shape, stddev = 0.1)

	return tf.Variable(weights)

def forward(X, w_1,w_2):
	"""forward-propagation
	Note: propagate inputs X through network
	"""
	h = tf.nn.sigmoid(tf.matmul(X, w_1))  # h: hidden layer (h = x * w_1 + b_1)
    y_hat = tf.matmul(h, w_2) # y_hat: estimated output (y_hat = h * w_2 + b_2)

    return y_hat

def concatenate_data(d1, d2):
	"""data pre-processing
	Note: concatenate two word embeddings, assuming d1 > d2 (if not, switch order) w/ same number of columns. 
	If a word exists in only one of embeddings, append a zero vector to the other embedding.
	"""
	result = pd.merge(govt, books, how='left', on='text')
	result.fillna(0, inplace=True)
	result.to_csv('result.csv') #(19929,201)

	text_col = result[result.columns[0]] # extract 'text' column
	conct_govt = result[result.columns[0:101]] #(19929, 101)
	conct_books = result[result.columns[101:]]
	conct_books.insert(loc=0, column='text', value=text_col) #(19929, 101)

	final_conct = np.concatenate((conct_govt, conct_books), axis=0) #(39858, 101)

	return final_conct

def split_data(data):
	"""read data set and split them into training and test set"""
	text_col = data[data.columns[0]] # extract 'text' column
	target = data[data.columns[101]] # extract last column (d_99; 100th dimension) ????

	# prepend column of 1s for bias
	r, c = data.shape
	all_X = np.ones((r, c+1))
	all_X[:, 1:] = data

	# change into one-hot vectors
	n_labels = len(np.unique(target))
	all_Y = np.eye(n_labels)[target]

	# split data into training/test sets for each X and Y
	return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def main():
	# load pre-processed/concatenated word embeddings
	input_data = concatenate_data(govt, books) #(39858, 101)
	n_instances = input_data.shape[0] #39858

	train_X, test_X, train_y, test_y = split_data(input_data)

	# set hidden layers' sizes
	x_size = train_X.shape[1] # number of input nodes 
	h_size = 500  # number of hidden nodes (in hidden layer)
	y_size = train_y.shape[1] # number of outcomes

	# set tf graph input 
	X = tf.placeholder("float", shape=[None, x_size])
	y = tf.placeholder("float", shape=[None, y_size])

	# set model weights
	w_1 = init_weight((x_size, h_size))
	w_2 = init_weight((h_size, y_size))

	# forward propagation
	y_hat = forward(X, w_1, w_2)
	pred = tf.argmax(y_hat, axis=1)

	# backward propagation
	cost =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
	update = tf.train.GradientDescentOptimizer(0.001).minimize(cost) # GD step size: 0.001

	# run stochastic gradient descent
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	for epoch in range(100):
		for i in range(len(train_X)):
			sess.run(update, feed_dict={X: train_X[i: i+1], y: train_y[i: i+1]})

		train_accuracy = np.mean(np.argmax(train_y, axis=1) == sess.run(pred, feed_dict={X: train_X, y: train_y}))
		test_accuracy = np.mean(np.argmax(test_y, axis=1) == sess.run(pred, feed_dict={X: test_X, y: test_y}))

		print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

	sess.close()

if __name__ == '__main__':
    main()
