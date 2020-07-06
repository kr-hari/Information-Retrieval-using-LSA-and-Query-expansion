from util import *
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD
import pdb  
from scipy.linalg import svd as scipy_svd
from scipy.sparse import save_npz, load_npz
import os
import numpy as np
import time
import joblib
from scipy.sparse.linalg import svds as svds
from nltk.corpus import wordnet
from util import query_expanded
from nltk.wsd import lesk
from util import *




class InformationRetrieval():
	def __init__(self,args):
		self.docstfidf = []
		self.querytfidf = []
		self.vector = []
		self.reduced_docs_tfidf, self.reduced_queries_tfidf = [], []
		self.index = 1
		self.docs, self.queries = [], []
		self.u, self.s, self.v_t = [], [], []
		self.vocabulary = []
		self.max_dim_lsa = args.max_dim 
		self.algo = args.method
		self.args = args

		self.countvectorizer = CountVectorizer(ngram_range = (1,args.max_ngram))
		self.transformer = TfidfTransformer()


	def vectorize_queries(self, queries):
		if self.algo == "Wordnet_QE" or self.algo == "Combined":
			print("Wordnet Weight : ", self.args.wordnet_weight)

		for query in queries:
			x = []
			for row in query:
				# SIMPLE QUERY EXPANSION
				if self.algo == "Wordnet_QE" or self.algo == "Combined":
					row = query_expanded(row, self.args.wordnet_weight)
				x.append(' '.join(map(str, row)))

			y = (' '.join(map(str, x)))
			self.queries.append(y)   

		# APPLY TF-IDF
		self.queriesvector = self.countvectorizer.transform(self.queries)
		self.queriesvector = self.transformer.transform(self.queriesvector).toarray()		

		print("Shape of New Query Vec : ",self.queriesvector.shape)

		if self.algo in ["Wordnet_QE","TFIDF"]:
			return self.docvector, self.queriesvector

		elif self.algo in ["LSA","Combined"]:
			self.queriesvector = np.matmul(self.queriesvector,self.projection_matrix)
			return np.matmul(self.s,self.D_t).transpose(),np.matmul(self.queriesvector, self.s)


	def wordnet(self,args,processed_corpus = []):

		self.processed_corpus = processed_corpus
		corpus = []
		for doc in processed_corpus:
			x = []
			for row in doc:
				x.append(' '.join(map(str, row)))
			y = (' '.join(map(str, x)))
			corpus.append(y)
		vocab = self.countvectorizer.fit(corpus)
		self.corpus = corpus
		vocab_vector = self.countvectorizer.transform(corpus)            
		vocab_vector = self.transformer.fit_transform(vocab_vector)
		self.docvector = vocab_vector[-1400:,:]



	def LSA(self, args, processed_corpus = []):    
		
		self.processed_corpus = processed_corpus
		
		import warnings
		warnings.filterwarnings("ignore", category=FutureWarning)

		print("Building vectorizer from corpus ...")
		
		corpus = []
		for doc in processed_corpus:
			x = []
			for row in doc:
				x.append(' '.join(map(str, row)))
			y = (' '.join(map(str, x)))
			corpus.append(y)

		vocab = self.countvectorizer.fit(corpus)
		self.corpus = corpus
		vocab_vector = self.countvectorizer.transform(corpus)            
		vocab_vector = self.transformer.fit_transform(vocab_vector)

		# Convert to Term - Document matrix
		vocab_vector = vocab_vector.transpose()

		# SVD of TF-IDF Matrix
		T,S,D_t = svds(vocab_vector.astype('float'), which = 'LM', k = min(min(vocab_vector.shape)-1,self.max_dim_lsa))
		S = np.diag(S)
		approx_dist(vocab_vector.astype('float'),  np.matmul(T,np.matmul(S,D_t)), rank = self.max_dim_lsa)

		self.D_t = D_t[:,-1400:]
		self.s =S

		self.projection_matrix = np.matmul(T,np.linalg.inv(S))

		self.doc_matrix = np.matmul(T,np.matmul(S,D_t))
		self.doc_matrix = self.doc_matrix[:,-1400:]
	
	



