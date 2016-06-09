import os
import glob
import sys
sys.path.insert(0,'../utils/')
from utils import utils
import numpy as np
from collections import Counter
import math


stop_words_dir = '../dat/stop_words.txt'

def getStopWords(fin):
    return set([term.strip() for term in fin])

def removeStopWords(d, stopwords):
    for term in set(d.keys()):
        if term in stopwords:
            del d[term]


# ========== FREQUENCIES ==========
#removePunct(x.lower()).encode('utf-8')

def getTermFrequencies(fin):
    tf = Counter()
    for line in fin: tf.update(map(lambda x: utils.removePunct(x.lower()), line.split()))
    return tf

def getDocumentFrequencies(tf):
    df = Counter()
    for d in tf: df.update(d.keys())
    return df

def getDocumentFrequency(df, term):
    if term in df: return df[term] + 1
    else: return 1

# ========== TF-IDF ==========

def getTFIDF(tf, df, dc):
    return tf * math.log(float(dc) / df)

def getTFIDFs(tf, df, dc):
    return [{k:getTFIDF(v, getDocumentFrequency(df,k), dc) for (k,v) in d.items()} for d in tf]


def getTFIDFScores(DIR, EXT):
    trnFiles = sorted(glob.glob(os.path.join(DIR,'*.'+EXT)))
    trnTF    = [getTermFrequencies(open(filename)) for filename in trnFiles]
    trnDF    = getDocumentFrequencies(trnTF)
    trnDC    = len(trnFiles) + 1
    trnTFIDF = getTFIDFs(trnTF, trnDF, trnDC)
    return trnTFIDF
    
    