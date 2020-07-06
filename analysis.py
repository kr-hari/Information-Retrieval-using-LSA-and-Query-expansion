"""
This code snippet runs LSA, Wordnet_WE and combined  algorithms and returns the evaluation metrics in each case
"""

from config import *
from main import * 

import argparse


if __name__ == '__main__':
	
	start_time = time.time()

	# Create an argument parser
	parser = argparse.ArgumentParser()

	# Tunable parameters as external arguments
	parser.add_argument('--dataset', default = DATASET_DIRECTORY, help = "Path to the dataset folder")
	parser.add_argument('--out_folder', default = "output/", help = "Path to output folder")
	parser.add_argument('--segmenter', default = "punkt",help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('--tokenizer',  default = "ptb", help = "Tokenizer Type [naive|ptb]")
	parser.add_argument("--corpus",type = str, default = "cranfield") #, choices = ["cranfield","20newsgroup","wiki_scrape", "combinedCWS","book_aerod4eng","book_flight"])
	parser.add_argument("--restrict_tokenization", type = str2bool, default = False, help = "Use only isalnum(True) or normal tokenization(False)")
	parser.add_argument("--max_dim", type = int, default = 600, help = "Maximum no of cordinates to be used" )
	parser.add_argument("--max_ngram", type = int, default = 1, help = "Maximum limit of Ngram. If 2 countvectorizer contains both unigrams and bigrams")
	args = parser.parse_args()

	if not  os.path.exists(args.out_folder):
		os.mkdir(args.out_folder)

	for args.method in ["LSA","Wordnet_QE","Combined","TFIDF"]:

		
		print("\nCorpus :",args.corpus, end = "\t")
		print("Restrictin Tokenization : ",args.restrict_tokenization, end = "\t")
		print("Maximum dimension : ",args.max_dim, end = "\t")
		print("Tokenizer : ",args.tokenizer, end = "\t")
		print("Segmenter : ",args.segmenter)

		# Create an instance of the Search Engine
		searchEngine = SearchEngine(args)
		searchEngine.evaluateDataset()

	print("Total time taken : ",time.time() - start_time)