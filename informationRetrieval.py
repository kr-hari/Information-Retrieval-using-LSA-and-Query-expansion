from util import *
import numpy as np



class InformationRetrieval():

	def __init__(self):
		self.index = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

###############       Inverted Index        #########################
		
		# Dictionary to map word to an index		
		word_dict = {}

		# variable to keep track of word index
		word_counter = 0

		# Dictionary to keep track of documents and their word frequency
		doc_info = {}
		# Vectorized form of <doc_info>
		doc_vec = {}

		# Find word frequencies in each document and save it using theor index
		# Format doc_info[<id of document>][<id of word>] = <Frequency of word in that document>
		for counter,doc in enumerate(docs):
			doc_id = docIDs[counter]
			doc_info[doc_id] = {}
			
			for sentence in doc:
				for word in sentence:
					if word not in word_dict:
						word_dict[word] = word_counter
						word_counter+=1
					word_id = word_dict[word]

					if word_id in doc_info[doc_id]:
						doc_info[doc_id][word_id] += 1
					else:
						doc_info[doc_id][word_id] = 1

			# Find the norm of the document (to save calculation later)
			doc_norm = np.linalg.norm(list(doc_info[doc_id].values()))

			# Create a normalized vector 
			doc_vec[doc_id] = []
			for key in sorted(doc_info[doc_id]):
				doc_vec[doc_id].append((key,doc_info[doc_id][key]/doc_norm))

		self.doc_vec = doc_vec
		self.word_dict = word_dict



	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""
		
		query_info = {}
		query_vec = {}

		# Find word frequencies in each query and save it using their index
		# Format query_info[<id of query>][<id of word>] = <Frequency of word in that query>
		for query_id,query in enumerate(queries):
			query_info[query_id] = {}
			for sentence in query:
				for word in sentence:
					try:
						word_id = self.word_dict[word]
						if word_id not in query_info[query_id]:
							query_info[query_id][word_id] = 1
						else:
							query_info[query_id][word_id] += 1
					except:
						continue

			# Find the norm of query
			query_norm = np.linalg.norm(list(query_info[query_id].values()))

			# Normalize each word frequency
			query_vec[query_id] = []
			for key in sorted(query_info[query_id]):
				query_vec[query_id].append((key, query_info[query_id][key]/query_norm))

		

		# List of list where each sub-list contain ids of document in their order of relevance 
		rank = []
		for q_id in query_vec:
			q_sim = {}
			for key in self.doc_vec:

				# Find cosine similarity of each document vector with query vector
				q_sim[key] = simple_cos_sim(query_vec[q_id],self.doc_vec[key])

			# Append ranks of each document for all queries
			rank.append([k for k,v in sorted(q_sim.items(), key=lambda item: item[1],reverse=True)])


		return rank
		