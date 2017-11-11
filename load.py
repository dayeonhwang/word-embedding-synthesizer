import csv
import random
import numpy as np
import pandas as pd

# Load input word embedding sets, concatenate them, fill missing values, and save it as "input.csv"

def read_csv_file(emb_path):
	"""read csv file
	"""
	emb_set = pd.read_csv(emb_path, header=0, encoding='utf-8')

	return emb_set

def concatenate_all_data(d):
	"""data pre-processing
	Note: concatenate k word embeddings on column "text"
	"""
	df = d[0]
	for d_next in d[1:]:
		if (df.shape[0] > d_next.shape[0]):
			df_new = pd.merge(df, d_next, how='outer', on='text')
			df = df_new
		else:
			df_new = pd.merge(d_next, df, how='outer', on='text')
			df = df_new

	return df

def main():
	# load word embeddings
	agri = read_csv_file('data/agriculture_40.csv')
	arts = read_csv_file('data/arts_40.csv')
	books = read_csv_file('data/books_40.csv')
	econ = read_csv_file('data/econ_40.csv')
	govt = read_csv_file('data/govt_40.csv')
	movies = read_csv_file('data/movies_40.csv')
	weather = read_csv_file('data/weather_40.csv')

	# concatenate 3 embeddings per iteration
	first_emb = [agri, arts, books]
	first_emb = concatenate_all_data(first_emb)
	second_emb = [econ, govt, movies]
	second_emb = concatenate_all_data(second_emb)
	final_emb = [first_emb, second_emb, weather]

	# concatenate k embeddings into 1 & replace empty values with random values
	input_emb = concatenate_all_data(final_emb)
	input_emb = input_emb[input_emb.columns[1:]]
	input_emb = input_emb.apply(lambda x: x.fillna(random.choice(x.dropna())), axis=1)

	# save the final input embedding set as a new csv file
	input_emb.to_csv('data/input.csv', index=False, encoding='utf-8')

	print("Final input embedding saved with size of %s at %s" % (str(input_emb.shape), 'data/input.csv'))

if __name__ == '__main__':
	main()