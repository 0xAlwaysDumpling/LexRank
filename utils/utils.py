import string
import os
import sys
from nltk import tokenize
import codecs
reload(sys)
sys.setdefaultencoding("utf-8")


def removePunct(s):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return s


def getSentences(doc_dir,doc_id):
	i = 0
	for subdir, dirs, files in os.walk(doc_dir):
		for file in files:
			filepath = subdir + file
			if i == doc_id:
				with open(filepath) as doc:
					sentences = tokenize.sent_tokenize(doc.read())
					return sentences
			else:
				i = i + 1 



def mapTFIDFScores(sentence, scores):
	words = sentence.split()
	mapping = dict((w,scores[removePunct(w.lower())]) for w in words)
	return mapping