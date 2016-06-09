import string
import sys
import os

sys.path.insert(0,'../vsm/')
sys.path.insert(0,'../functions/')
sys.path.insert(0,'../utils/')
from utils import utils
from vsm import tfidf
import numpy as np
import fnmatch


class LexRank(object):
	def __init__(self, docs_dir, sentence_threshold, ext='txt'):
		self.doc_dir = docs_dir
		self.tfidf_scores = tfidf.getTFIDFScores(docs_dir, ext)
		self.sentence_threshold = sentence_threshold
		

	def singleDocSummarization(self, doc_id, similarityFunct, normalizeFunct):
		self.sentences = utils.getSentences(self.doc_dir,doc_id)
		self.num_of_sentences = len(self.sentences)
		self.matrix = self.buildSimilarityMatrix(self.num_of_sentences, self.sentences, doc_id, similarityFunct, normalizeFunct)

	def multiDocSummarization(self, similarityFunct, normalizeFunct):
		matrices = []
		sentences = []
		num_of_docs = len(fnmatch.filter(os.listdir(self.doc_dir), '*.txt'))
		for x in range(0,num_of_docs):
			curr_sentences = utils.getSentences(self.doc_dir,x)
			curr_num_of_sentences = len(curr_sentences)
			sentences.append(curr_sentences)
			matrices.append(self.buildSimilarityMatrix(curr_num_of_sentences, curr_sentences, x, similarityFunct, normalizeFunct))
		self.matrix = matrices
		self.sentences = sentences

	def buildSimilarityMatrix(self, num_of_sentences, sentences, doc_id, similarityFunct, normalizeFunct):
		matrix = np.zeros((num_of_sentences, num_of_sentences))
		for x in range(0, num_of_sentences):
			vec_x = utils.mapTFIDFScores(sentences[x], self.tfidf_scores[doc_id])
			for y in range(0,num_of_sentences):
				if x == y:
					matrix[x][y] = 1.0
					continue
				vec_y = utils.mapTFIDFScores(sentences[y], self.tfidf_scores[doc_id])
				score = similarityFunct(vec_x, vec_y)
				if np.isnan(score): score = 0.0
				matrix[x][y] = score

		#Normalize each row of scores
		for x in range(0, num_of_sentences):
			matrix[x] = normalizeFunct(matrix[x], num_of_sentences)
		return matrix
	
	


# class LexRank(object):
# 	def __init__(self, doc_dir, sentence_threshold, doc_id = 1, ext='txt', num_of_sentences = 0, sentences = None):
# 		self.tfidf_scores = tfidf.getTFIDFScores(doc_dir,ext)
# 		self.doc_id = doc_id
# 		self.sentence_threshold = sentence_threshold
# 		if num_of_sentences > 0:
# 			self.sentences = sentences
# 			self.num_of_sentences = num_of_sentences
# 		else:
# 			self.sentences = utils.getSentences(doc_dir,doc_id)
# 			self.num_of_sentences = len(self.sentences)
# 		self.matrix = np.zeros((self.num_of_sentences,self.num_of_sentences))



# 	def buildSimilarityMatrix(self, similarityFunct, normalizeFunct):
# 		for x in range(0,self.num_of_sentences):
# 			vec_x = utils.mapTFIDFScores(self.sentences[x], self.tfidf_scores[self.doc_id])
# 			for y in range(0,self.num_of_sentences):
# 				if x == y:
# 					self.matrix[x][y] = 1.0
# 					continue
# 				vec_y = utils.mapTFIDFScores(self.sentences[y], self.tfidf_scores[self.doc_id])
# 				score = similarityFunct(vec_x, vec_y)
# 				if np.isnan(score): score = 0.0
# 				self.matrix[x][y] = score

# 		#Normalize each row of scores
# 		for x in range(0, self.num_of_sentences):
# 			self.matrix[x] = normalizeFunct(self.matrix[x], self.num_of_sentences)





