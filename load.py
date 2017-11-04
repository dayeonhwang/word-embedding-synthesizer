import random
import numpy as np
import pandas as pd

# Load input word embedding sets, concatenate them, fill missing values, and save it as "input.csv"

def read_csv(emb_path):
	"""read csv file
	"""
	emb_set = pd.read_csv(emb_path, encoding='ISO-8859-1')
	return emb_set

def concatenate_all_data(d):
	"""data pre-processing
	Note: concatenate more than two word embeddings, by using the helper function "concatenate_two_data"
	"""
	d_tmp = concatenate_two_data(d[0], d[1])
	if (len(d) == 2):
		return d_tmp
	elif (len(d) > 2):
		for i in range(2,len(d)-1):
			d_tmp = concatenate_two_data(d_tmp, d[i])
		return d_tmp

def concatenate_two_data(d1, d2):
	"""data pre-processing
	Note: concatenate two word embeddings, assuming d1 > d2 (if not, switch order) & same number of columns.
	Dimension of final dataset will be same as d1. 
	If a word exists in only one of embeddings, fill empty cell w/ random non-empty value from same dataset.
	"""
	if (d2.shape[0] > d1.shape[0]):
		concatenate_data(d2, d1)
	else: 
		result = pd.merge(d1, d2, how='left', on='text')
		result = result[result.columns[1:]]
		result = result.apply(lambda x: x.fillna(random.choice(x.dropna())), axis=1)

	return result

def main():
	# load word embeddings
	govt = read_csv('data/govt_40.csv') #(19928, 101)
	books = read_csv('data/books_40.csv') #(17506,101)
	k_embeddings = [govt, books]

	# concatenate k embeddings into 1
	input_emb = concatenate_all_data(k_embeddings) #(19929,201)

	# replace NaN(empty) values with random value
	input_emb.to_csv('data/input.csv', index=False)


if __name__ == '__main__':
	main()