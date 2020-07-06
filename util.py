from config import * 

import json
import os
import pdb
import matplotlib.pyplot as plt
import glob
from nltk.corpus import wordnet
from nltk.wsd import lesk

def load_cranfield_docs():
	"""
	Function to read Cranfield document data
	"""
	docs_json = json.load(open(os.path.join(CRANFIELD_DIRECTORY,"cran_docs.json"), 'r'))[:]
	doc_ids, docs_ = [item["id"] for item in docs_json], \
								[item["body"] + item["title"] + item["title"] for item in docs_json]
	return docs_

def get_corpus(corpus = "20newsgroup"):
	"""
	Function to return the desired corpus for training. Make sure the paths are correct.
	Cranfield dataset is appended by default in some cases.
	"""
	if corpus == "20newsgroup":
		from sklearn.datasets import fetch_20newsgroups as ng
		corpus = ng().data
		return corpus 

	elif corpus == "cranfield":		
		return load_cranfield_docs()			

	elif corpus == "books":
		docs = []
		for file in glob.glob(os.poth.join(PATH_TO_BOOKS_FOLDER,r"\*.pdf")):
			try:
				docs.extend(convert_book_to_pdf(file))
			except:
				pass				
		docs.extend(load_cranfield_docs())

	else:
		print("Corpus not found. Exiting ...")
		return

	print("Length of corpus : ", len(docs))
	return docs


def convert_book_to_pdf(filename):
	"""
	Function to convert a book (in pdf) to list of texts
	"""
	import PyPDF2 
	import textract 
	from nltk.tokenize import word_tokenize
	from nltk.corpus import stopwords

	#The pdfReader variable is a readable object that will be parsed.
	pdfFileObj = open(filename,'rb')
	#Discerning the number of pages will allow us to parse through all the pages.
	pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
		
	if pdfReader.isEncrypted:
		try:
			pdfReader.decrypt("")
			print("File Decrypted")
		except:
			command="copy "+filename+" temp.pdf; qpdf --password='' --decrypt temp.pdf "+filename
			os.system(command)
			print('File Decrypted (qpdf)')
			#re-open the decrypted file
			fp = open(filename,'rb')
			pdfReader = PyPDF2.PdfFileReader(fp)

		
	num_pages = pdfReader.numPages
	count = 0	
	pages = []
	while count < num_pages:
		pageObj = pdfReader.getPage(count)
		count +=1

		#This if statement exists to check if the above library returned words. It's done because PyPDF2 cannot read scanned files.
		text = pageObj.extractText()
		pages.append(text)
	return pages

def simple_cos_sim(doc_vec,query_vec):
	"""
	Function to calculate cosine similarity between two vectors in case of sparse matrix when the input file is given as a list of tuple
	where each tuple indicate the word id and its normalized frequency
	eg: dic_vec = [(5,0.13),(26,0.06)]   (Similarly query vec)
	"""
	i = 0; j = 0
	cos_sim = 0
	while i < len(doc_vec) and j < len(query_vec):
		if doc_vec[i][0] == query_vec[j][0]:
			cos_sim += doc_vec[i][1] * query_vec[j][1]
			i += 1
			j += 1
		elif doc_vec[i][0] < query_vec[j][0]:
			i += 1
		else:
			j += 1
	
	return cos_sim


def get_true_doc_IDs(qrels):
	"""
		Function to return a dictionary of True_document_IDs where each key is a query_id 
		and each value represents the list of ids of documents relevant to that query
	"""

	true_doc_IDs = {}
	for qrel in qrels:
		q_num = int(qrel["query_num"])
		if q_num in true_doc_IDs:
			true_doc_IDs[q_num].append(int(qrel["id"]))
		else:
			true_doc_IDs[q_num] = []
			true_doc_IDs[q_num].append(int(qrel["id"]))
	return true_doc_IDs

def get_true_id_with_rank(qrels):
	"""
		Function to return a dictionary of True_document_IDs where each key is a query_id 
		and each value represents a dictionary with keys as id of relevant documents and values
		as their rank (position as given in qrel)
	"""
	ranked_true_ids = {}
	for qrel in qrels:
		q_num = int(qrel["query_num"])
		if q_num in ranked_true_ids:
			ranked_true_ids[q_num][int(qrel["id"])] = int(qrel["position"])
		else:
			ranked_true_ids[q_num] = {}
			ranked_true_ids[q_num][int(qrel["id"])] = int(qrel["position"])
	
	return ranked_true_ids

def str2bool(v):
	if isinstance(v, bool):
	   return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_eig(s, args): 
	"""
	Function to plot Variance captured for each eigen value
	"""
	plt.plot([i*i for i in s[::-1]])
	plt.xlabel('Number of components')
	plt.ylabel('Variance * (N-1) ')    # Where N is the sample size of corpus
	plt.title('Variance explained by components')
	plt.savefig(args.out_folder + args.corpus + " Component_Variance.png")

def approx_dist(orig, approimation, rank):
	"""
	Function to calculate euclidean distance (L2 norm) between two matrices of same size.
	"""
	import numpy as np 

	diff = orig - approimation
	print("Rank = ",rank,"\t Euclidean Distance = ",np.sqrt(np.sum(np.square(diff))))

def query_expanded(list_of_words, weight, word_similarity = None):
	"""
	Expand the query using various word relations (synonyms, hypernyms or hyponyms)
	"""
	count = 0
	expanded_query = []
	for x in list_of_words:	
		expanded_query.extend([x for i in range(weight)])
		# # WSD
		syn = lesk(list_of_words, x)
		try:
			for l in syn.lemmas() :
				# if(count<3):
				if l.name() not in expanded_query:
					expanded_query.append(l.name())
					count+=1
			# for hyp in syn.hypernyms():
			# 	for hyp_lemma in hyp.lemmas():
			# 		if hyp_lemma.name() not in expanded_query:
			# 			expanded_query.append(hyp_lemma.name())
			# 			count+=1
			# for hyp in syn.hyponyms():
			# 	for hyp_lemma in hyp.lemmas():
			# 		if hyp_lemma.name() not in expanded_query:
			# 			expanded_query.append(hyp_lemma.name())
			# 			count+=1
		except:
			pass

	return expanded_query

def explain_queries(queries,qrels,doc_IDs_ordered,queries_json,docs_json):
	""" 
	Function to print and verify the expected and predicted results of desired queries 

	queries : Queries (id) to be considered
	qrels	   	 : Expected results
	doc_IDs_ordered : Obtained results from model
	"""
	
	poor_queries.sort(key = lambda x:x[1], reverse=True)
	print("Poor Quries : ", poor_queries)
	ranked_qrels  = get_true_id_with_rank(qrels)

	query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]

	doc_ids, docs = [item["id"] for item in docs_json], \
								[item["body"] for item in docs_json]
	for query_id, score in poor_queries[-5:]:

		query = queries[query_id-1]
		print("Query : \"",query,"\" \t\t nDCG :", score, "\t Query id : ",query_id)
		expected_output_ids = sorted(ranked_qrels[query_id].items(),key = lambda kv:(kv[1], kv[0]))[:3]
		obtained_output_ids = doc_IDs_ordered[query_id-1][:3]

		print ("******RETRIEVED DOCUMENTS******")
		for doc_id in  obtained_output_ids:
			print("Doc ID : ",doc_id)
			print(docs[doc_id-1])
			print()

		print ("******EXPECTED DOCUMENTS******")
		for doc_id,rank in  expected_output_ids:
			print("Doc ID : ",doc_id)
			print("Relevance score (Max 1) : ",rank)
			print(docs[doc_id-1])
			print()

		print("\n\n\n")
