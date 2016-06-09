import string
import sys
import os

sys.path.insert(0,'../vsm/')
sys.path.insert(0,'../functions/')
sys.path.insert(0,'../utils/')
from utils import utils
from vsm import tfidf
import numpy as np


class LR(object):
	def __init__(self, docs_dir, sentence_threshold, ext='txt'):
		self.tfidf_scores = tfidf.getTFIDFScores(docs_dir, ext)
		self.sentence_threshold = sentence_threshold
		

	def singleDocSummarization(self, doc_id, similarityFunct, normalizeFunct):
		pass

	def multiDocSummarization(self, doc_id, similarityFunct, normalizeFunct):
		pass


class LexRank(object):
	def __init__(self, doc_dir, sentence_threshold, doc_id = 1, ext='txt', num_of_sentences = 0, sentences = None):
		self.tfidf_scores = tfidf.getTFIDFScores(doc_dir,ext)
		self.doc_id = doc_id
		self.sentence_threshold = sentence_threshold
		if num_of_sentences > 0:
			self.sentences = sentences
			self.num_of_sentences = num_of_sentences
		else:
			self.sentences = utils.getSentences(doc_dir,doc_id)
			self.num_of_sentences = len(self.sentences)
		self.matrix = np.zeros((self.num_of_sentences,self.num_of_sentences))



	def buildSimilarityMatrix(self, similarityFunct, normalizeFunct):
		for x in range(0,self.num_of_sentences):
			vec_x = utils.mapTFIDFScores(self.sentences[x], self.tfidf_scores[self.doc_id])
			for y in range(0,self.num_of_sentences):
				if x == y:
					self.matrix[x][y] = 1.0
					continue
				print len(self.tfidf_scores[1])
				vec_y = utils.mapTFIDFScores(self.sentences[y], self.tfidf_scores[self.doc_id])
				score = similarityFunct(vec_x, vec_y)
				if np.isnan(score): score = 0.0
				self.matrix[x][y] = score

		#Normalize each row of scores
		for x in range(0, self.num_of_sentences):
			self.matrix[x] = normalizeFunct(self.matrix[x], self.num_of_sentences)





