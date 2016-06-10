import string
import sys
import os
import math
import numpy as np
import fnmatch
sys.path.insert(0,'../vsm/')
sys.path.insert(0,'../functions/')
sys.path.insert(0,'../utils/')
from utils import utils
from vsm import tfidf



class LexRank(object):
	def __init__(self, docs_dir, ext='txt'):
		self.doc_dir = docs_dir
		self.tfidf_scores = tfidf.getTFIDFScores(docs_dir, ext)


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
	
	def rank(self, similarity_threshold, epsilon, iterations, continuous, damp, multi=False):
		if multi:
			return multiDocRank(self.matrix,self.sentences, similarity_threshold, epsilon, iterations, continuous, damp)
		else:
			return singleDocRank(self.matrix, self.sentences, similarity_threshold, epsilon, iterations, continuous, damp)




def multiDocRank(matrices, sentences, similarity_threshold, epsilon, iterations, continuous, damp):
	rankings = []
	for x in range(0, len(matrices)):
		num_of_sentences = len(matrices[x][0])
		similarity_matrix = reflectOverYX(matrices[x], num_of_sentences)
		transition_matrix = buildTransitionMatrix(similarity_matrix, num_of_sentences, similarity_threshold, continuous)
		rank,score = powerIteration(transition_matrix, num_of_sentences, epsilon, iterations, damp)
		rankings.append(rank)
	return rankings,sentences


def singleDocRank(matrix, sentences, similarity_threshold, epsilon, iterations, continuous, damp):
	num_of_sentences = len(matrix[0])
	similarity_matrix = reflectOverYX(matrix, num_of_sentences)
	transition_matrix = buildTransitionMatrix(similarity_matrix, num_of_sentences, similarity_threshold, continuous)
	
	rankings,score = powerIteration(transition_matrix, num_of_sentences, epsilon, iterations, damp)
	return rankings,sentences




def powerIteration(stochastic_matrix, num_of_sentences, epsilon, maxIterations, damp):
    if damp > 0:
       curr_matrix = applyDampingFactor(stochastic_matrix, num_of_sentences,damp).transpose()
    else:
        curr_matrix = stochastic_matrix.transpose()
    curr_vec = np.zeros((num_of_sentences,1))
    prev_vec = np.zeros((num_of_sentences,1))
    curr_vec.fill(1.0/num_of_sentences)

    for x in range(0,maxIterations):
        prev_vec = curr_vec
        curr_vec = np.dot(curr_matrix,curr_vec)
        error = 0.0
        for y in range(0, num_of_sentences):
            error += math.sqrt(abs(curr_vec[y][0]-prev_vec[y][0]))
        if error < math.sqrt(epsilon):
            break

    #Scores sorted from lowest to biggest
    scores = [curr_vec[x][0] for x in range(0,num_of_sentences)]
    

    #Ranking based on biggest score to lowest so indexes are reversed
    rankings = np.argsort(scores)
    rankings = rankings[::-1]
    return rankings,scores


def applyDampingFactor(matrix, N, d):
    #1-dB
    matrix = np.multiply(matrix,1-d)
    dU = np.zeros((N,N))
    dU.fill(1.0/N)
    return dU + matrix

def findNeighbors(transition_matrix, num_of_sentences):
    n_map = dict()
    for x in range(0, num_of_sentences):
        for y in range(0, num_of_sentences):
            if transition_matrix[x][y] > 0:
                if x in n_map:
                    l = n_map[x]
                    l.append(y)
                else:
                    l = []
                    l.append(y)
                    n_map[x] = l
    return n_map




#Build transition probability matrix with threshold | continuous
def buildTransitionMatrix(similarity_matrix, num_of_sentences, similarity_threshold = .1, continuous=True):
    transition_matrix = np.zeros((num_of_sentences,num_of_sentences))
    for x in range(0,num_of_sentences):
        degree = 0
        for y in range(0, num_of_sentences):
            if similarity_matrix[x][y] > similarity_threshold:
                if continuous:
                    transition_matrix[x][y] = similarity_matrix[x][y]
                    degree += similarity_matrix[x][y]
                else:
                    transition_matrix[x][y] = 1
                    degree += 1
        for y in range(0,num_of_sentences):
            transition_matrix[x][y] /= degree

    return transition_matrix




#Reflect Scores of the matrix over the line y = x b/c index order of sentence doesn't matter
def reflectOverYX(matrix, num_of_sentences):
    result = np.zeros((num_of_sentences, num_of_sentences))
    for x in range(0,num_of_sentences):
        for y in range(0,x+1):
            result[x][y] = result[y][x] = matrix[x][y]
    return result
		




