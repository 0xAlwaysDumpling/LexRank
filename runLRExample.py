import os 
import sys
from nltk import tokenize
from core import LexRank as l
from functions import similarity
from functions import normalize


#DOC_NUM is 0 indexed
DOC_PATH = './data/sample_dat/'
DOC_NUM = 0
EXT = 'txt'

# rank our sentences
cosine_threshold = 0.1
epsilon = 0.005
iterations = 100
continuous_flag = True
damping_factor = 0.15



lr = l.LexRank(DOC_PATH, EXT)
lr.singleDocSummarization(DOC_NUM, similarity.cosine, normalize.normalizeByLength)
rankings, sentences = lr.rank(cosine_threshold, epsilon, iterations, continuous_flag, damping_factor, False)

for s in rankings[0:5]:
    print sentences[s]