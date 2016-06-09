

from nltk import tokenize
from textblob import TextBlob
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
with open('../data/sample_dat/barack.txt', 'rb') as doc:
	f = TextBlob(doc.read())
	print f.sentences
