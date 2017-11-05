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
	Note: concatenate k word embeddings on column "text"
	"""

	df = d[0]
	for d_next in d[1:]:
		if (df.shape[0] > d_next.shape[0]):
			df = pd.merge(df, d_next, how='left', on='text')
		else:
			df = pd.merge(d_next, df, how='left', on='text')

	return df

def main():
	# load word embeddings
	agri = read_csv('data/agriculture_40.csv')
	arts = read_csv('data/arts_40.csv')
	books = read_csv('data/books_40.csv') #(17506,101)
	econ = read_csv('data/econ_40.csv')
	govt = read_csv('data/govt_40.csv') #(19928, 101)
	movies = read_csv('data/movies_40.csv')
	weather = read_csv('data/weather_40.csv')

	k_embeddings = [agri, arts, books, econ, govt, movies, weather]

	# concatenate k embeddings into 1 & replace empty values with random values
	input_emb = concatenate_all_data(k_embeddings) #(19929,201)
	input_emb = input_emb[input_emb.columns[1:]]
	input_emb = input_emb.apply(lambda x: x.fillna(random.choice(x.dropna())), axis=1)

	# save the final input embedding set as a new csv file
	input_emb.to_csv('data/input.csv', index=False)

	print("Final input embedding saved with size of %s at %s" % (str(input_emb.shape), 'data/input.csv'))


if __name__ == '__main__':
	main()