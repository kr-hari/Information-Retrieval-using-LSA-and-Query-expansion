from util import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances as pd
from scipy.spatial.distance import jensenshannon as js
import numpy as np
import pdb

class Similarity():
    """
    Class to finc the similarity between evaluation metric
    """
    def __init__(self):
        self.sim = []
        self.rank = []


    def find_similarity(self, doc_tfidf, query_tfidf, method):
        print("Shape of D_q : ",query_tfidf.shape)
        print("Shape of D_t : ",doc_tfidf.shape)
        if method == "lsa" or method == "combined":
            
            #########  LSA similarity
            self.sim = np.matmul(query_tfidf, doc_tfidf)

            #########  Euclidian distance 
            # self.sim = pd(query_tfidf,doc_tfidf.transpose())
            # for i in range(self.sim.shape[0]):
            #     for j in range(self.sim.shape[1]):
            #         self.sim[i][j] = 1/(self.sim[i][j] + 0.001)

            #########  Jensen Shannon Distance
            # doc_tfidf = doc_tfidf.transpose()
            # self.sim = [[0 for i in range(doc_tfidf.shape[0])] for j in range(query_tfidf.shape[0])]
            # for i in range(query_tfidf.shape[0]):
            #     for j in range(doc_tfidf.shape[0]):
            #         self.sim[i][j] = js(query_tfidf[i], doc_tfidf[j])
        else:
            #########  Cosine Similarity
             self.sim = cosine_similarity(query_tfidf, doc_tfidf)

    def Rank(self):
        for i in self.sim:
            self.rank.append(sorted(range(len(i)), key=lambda item: i[item],reverse=True))
            self.rank[-1] = [j+1 for j in self.rank[-1]]
        return self.rank