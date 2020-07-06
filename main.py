from information import InformationRetrieval
from similarity import Similarity
from evaluation import Evaluation
from preprocessing import Preprocessing
from util import * 
from config import *

from sys import version_info
import argparse
import json
import pdb
import numpy as np
import time

class SearchEngine:

	def __init__(self, args):
		self.args = args
		self.preprocess = Preprocessing()
		self.informationRetriever = InformationRetrieval(self.args)
		self.similarity = Similarity()
		self.evaluator = Evaluation()


	def preprocessQueries(self, queries):

		################# Segment queries ################

		segmentedQueries = []
		for query in queries:
			segmentedQuery = self.preprocess.sentenceSegmentation(query)
			segmentedQueries.append(segmentedQuery)
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))

		################## Tokenize queries #################

		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQuery = self.preprocess.tokenization(query, self.args)
			tokenizedQueries.append(tokenizedQuery)
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
		
		################### Stem/Lemmatize queries #################

		reducedQueries = []
		for query in tokenizedQueries:
			reducedQuery = self.preprocess.reduction(query)
			reducedQueries.append(reducedQuery)
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
		
		#################### Remove stopwords from queries #################

		stopwordRemovedQueries = []
		for query in reducedQueries:
			stopwordRemovedQuery = self.preprocess.stopwordremove(query)
			stopwordRemovedQueries.append(stopwordRemovedQuery)
		json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

		preprocessedQueries = stopwordRemovedQueries
		return preprocessedQueries


	def preprocessDocs(self, docs):

		####################### Segment docs #########################
		# 

		segmentedDocs = []
		for doc in docs:
			segmentedDoc = self.preprocess.sentenceSegmentation(doc)
			segmentedDocs.append(segmentedDoc)
		json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
		
		####################### Tokenize docs ##########################

		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDoc = self.preprocess.tokenization(doc,self.args)
			tokenizedDocs.append(tokenizedDoc)
		json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))

		####################### Stem/Lemmatize docs ########################

		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDoc = self.preprocess.reduction(doc)
			reducedDocs.append(reducedDoc)
		json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
		
		########################## Remove stopwords from docs #############################

		stopwordRemovedDocs = []
		for doc in reducedDocs:
			stopwordRemovedDoc = self.preprocess.stopwordremove(doc)
			stopwordRemovedDocs.append(stopwordRemovedDoc)
		json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

		preprocessedDocs = stopwordRemovedDocs
		return preprocessedDocs

	def evaluateDataset(self):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""
		args =self.args

		########################## Read queries ##########################

		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
								
		########################## Process queries ########################## 

		processedQueries = self.preprocessQueries(queries)

		# ########################## Read documents ##########################

		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]

		########################## BUILD DATASET FOR LSA ###########################

		if args.method in ["Wordnet_QE","TFIDF"]:
			corpus = get_corpus(corpus=args.corpus)
			processed_corpus = self.preprocessDocs(corpus)
			self.informationRetriever.wordnet(args,processed_corpus)

		elif args.method == "LSA" or args.method == "Combined":			
			corpus = get_corpus(corpus=args.corpus)
			processed_corpus = self.preprocessDocs(corpus)
			self.informationRetriever.LSA( args, processed_corpus)
			

		########################## Build document index ##########################

		doc_tfidf, query_tfidf = self.informationRetriever.vectorize_queries(processedQueries)

		###################### Rank the documents for each query ##########################

		self.similarity.find_similarity(doc_tfidf, query_tfidf, method = self.args.method)
		doc_IDs_ordered = self.similarity.Rank()

		###################### Read relevance judements ##########################

		qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

		######## Calculate precision, recall, f-score, MAP and nDCG for k ##########

		precisions, recalls, fscores, MAPs, nDCGs, x = [], [], [], [], [], []
		
		for k in range(1, 11, 1):
			x.append(k)
			precision = self.evaluator.meanPrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			precisions.append(precision)
			recall = self.evaluator.meanRecall(
				doc_IDs_ordered, query_ids, qrels, k)
			recalls.append(recall)
			fscore = self.evaluator.meanFscore(
				doc_IDs_ordered, query_ids, qrels, k)
			fscores.append(fscore)
			MAP = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			MAPs.append(MAP)
			nDCG, query_scores = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, k)
			nDCGs.append(nDCG)
		print(" Following metrics are averaged over k = 1 to 10 ")
		print(" nDCG : ", np.round(np.mean(nDCGs),3), end = "\t")
		print(" MAP : ", np.round(np.mean(MAPs),3), end = "\t")
		print(" Precision : ", np.round(np.mean(precisions),3), end = "\t")
		print(" Recall : ", np.round(np.mean(recalls),3), end = "\t")
		print(" F-score : ",np.round( np.mean(fscores),3), end = "\t")
		print("\n For k = 1 ")
		print(" nDCG : ", np.round(nDCGs[0],3), end = "\t")
		print(" MAP : ", np.round(MAPs[0],3), end = "\t")
		print(" Precision : ", np.round(precisions[0],3), end = "\t")
		print(" Recall : ", np.round(recalls[0],3), end = "\t")
		print(" F-score : ",np.round( fscores[0],3)	)

		############# Plot the metrics and save plot ######################

		plt.plot(x, precisions, label="Precision")
		plt.plot(x, recalls, label="Recall")
		plt.plot(x, fscores, label="F-Score")
		plt.plot(x, MAPs, label="MAP")
		plt.plot(x, nDCGs, label="nDCG")
		plt.legend()
		plt.title("Evaluation Metrics - Cranfield Dataset")
		plt.xlabel("k")
		plt.savefig(args.out_folder + "eval_plot.png")

		return [query_scores,doc_IDs_ordered, nDCGs, MAPs, precisions, recalls, fscores]



if __name__ == "__main__":

	start_time = time.time()

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('--dataset', default = DATASET_DIRECTORY, help = "Path to the dataset folder")
	parser.add_argument('--out_folder', default = "output/", help = "Path to output folder")
	parser.add_argument('--segmenter', default = "punkt",help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('--tokenizer',  default = "ptb",help = "Tokenizer Type [naive|ptb]")
	parser.add_argument("--corpus",type = str, default = "cranfield", help= "Corpus used for training")
	#choices = ["cranfield","20newsgroup","books"])
	parser.add_argument("--restrict_tokenization", type = str2bool, default = False, help = "Use only isalnum(True) or normal tokenization(False)")
	parser.add_argument("--max_dim", type = int, default = 1000, help = "Maximum no of cordinates to be used (LSA)" )
	parser.add_argument("--max_ngram", type = int, default = 1, help = "Maximum limit of Ngram. If 2 countvectorizer contains both unigrams and bigrams")
	parser.add_argument("--method", type = str, default = 'LSA', help = "Algorithm for IR", choices =['LSA','Wordnet_QE','Combined','TFIDF'])
	# Wordnet_QE => Query expansion based on WordNet
	# LSA 		 => Latent Semantic Analysis
	# Combined   => Combined LSA + Wordnet_QE	
	parser.add_argument("--wordnet_weight", type = int, default = 1, help = "Weight of actual word compared to appended words (WordNet_QE)" )
	
	args = parser.parse_args()
	print("\nCorpus :",args.corpus, end = "\t")
	print("Restrictin Tokenization : ",args.restrict_tokenization, end = "\t")
	print("Maximum dimension : ",args.max_dim, end = "\t")
	print("Tokenizer : ",args.tokenizer, end = "\t")
	print("Segmenter : ",args.segmenter, end = "\t")
	print("Algorithm : ",args.method)

	if not  os.path.exists(args.out_folder):
		os.mkdir(args.out_folder)

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	searchEngine.evaluateDataset()

	print("Total time taken : ",time.time() - start_time)
	# Run  - python main.py --corpus=cranfield  --method=wordnet2 --use_svd=True --restrict_tokenization=False --max_dim=600 --max_ngram=1 --top_n=1
