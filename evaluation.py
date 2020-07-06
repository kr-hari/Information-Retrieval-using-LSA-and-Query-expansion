from util import *
import numpy as np

class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""
		true_doc_IDs = set(true_doc_IDs)
		precision = -1
		tp = 0
		for i in range(k):
			if int(query_doc_IDs_ordered[i]) in true_doc_IDs:
				tp += 1
		precision = tp/k
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""
		meanPrecision = -1
		true_doc_IDs = get_true_doc_IDs(qrels)
		precision = []
		for query_id in query_ids:
			precision.append(self.queryPrecision(doc_IDs_ordered[query_id - 1], query_ids[query_id - 1], true_doc_IDs[query_id], k))

		meanPrecision = np.mean(precision)

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""
		true_doc_IDs = set(true_doc_IDs)
		recall = -1
		tp = 0
		for i in range(k):
			if int(query_doc_IDs_ordered[i]) in true_doc_IDs:
				tp += 1
		recall = tp/len(true_doc_IDs)

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		true_doc_IDs = get_true_doc_IDs(qrels)

		recall = []
		for query_id in query_ids:	
			recall.append(self.queryRecall(doc_IDs_ordered[query_id - 1], query_ids[query_id - 1], true_doc_IDs[query_id], k))

		meanRecall = np.mean(recall)

		#Fill in code here

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if (precision + recall) == 0.0:
			fscore = 0
		else:
			fscore = 2 * precision * recall / (precision + recall)


		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1
		Fscore = []
		true_doc_IDs = get_true_doc_IDs(qrels)
		for query_id in query_ids:	
			# true_doc_IDs = get_true_doc_IDs(qrels, query_id)
			Fscore.append(self.queryFscore(doc_IDs_ordered[query_id - 1], query_ids[query_id - 1], true_doc_IDs[query_id], k))

		meanFscore = np.mean(Fscore)

		return meanFscore

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""
		DCG = 0.0
		IDCG = 0.0

		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				relevance = self.position_value[true_doc_IDs.index(query_doc_IDs_ordered[i])]
				DCG += relevance/np.log2(i+2)

			if i<len(self.position_value):
				IDCG += self.position_value[i]/np.log2(i+2)

		nDCG = DCG/IDCG

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""
		nDCG = []
		queries = []
		self.ranked_true_docs = get_true_id_with_rank(qrels)

		for query_id in query_ids:
			relevant_docs = []
			self.position_value = []
			for doc_ID,position in sorted(self.ranked_true_docs[query_id].items(), key=lambda item: item[1]):
				relevant_docs.append(doc_ID)
				# Higher value for lower rank, i.e position 1 will have a position value of 4.
				self.position_value.append(5-position)
			nDCG_value = self.queryNDCG(doc_IDs_ordered[query_id - 1],query_id,relevant_docs,k)
			nDCG.append(nDCG_value)

			queries.append((query_id,nDCG_value))

		meannDCG = np.mean(nDCG)

		return meannDCG, queries


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1
		precision = []
		for i in range(k):
			precision.append(self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i+1))
		avgPrecision = np.mean(precision)

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		true_doc_IDs = get_true_doc_IDs(q_rels)

		AveragePrecision = []
		for query_id in query_ids:	
			AveragePrecision.append(self.queryAveragePrecision(doc_IDs_ordered[query_id - 1], query_ids[query_id - 1], true_doc_IDs[query_id], k))

		meanAveragePrecision = np.mean(AveragePrecision)

		return meanAveragePrecision

