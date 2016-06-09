import os 
import sys
from nltk import tokenize
from core import LexRank as l
from functions import similarity
from functions import normalize


#from functions import normalize, rank, similarity

#DOC_NUM is 0 indexed
DOC_PATH = './data/sample_dat/'
SENTENCE_CUTOFF = 5
DOC_NUM = 0
EXT = 'txt'
lr = l.LexRank(DOC_PATH, SENTENCE_CUTOFF, DOC_NUM, EXT, 0, None)
lr.buildSimilarityMatrix(similarity.cosine, normalize.normalizeByLength)


