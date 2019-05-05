
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

from nltk.corpus import reuters
import re
import numpy as np
import pandas as pd
from pprint import pprint
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import lda

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import utils

# spacy for lemmatization
#import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
    
# NLTK Stop words
from nltk.corpus import stopwords


#reuters.categories()
    

documents = reuters.fileids()

# Documents in a category
category_docs = reuters.fileids("jet");




corpus = []
corpus_tokenize = []
for i in range(0, int(len(documents)/100)):#10788
   # word_tokenize = word_tokenize(document_words[i])
   # document_words = list(reuters.words(category_docs[i]))
    document_words = list(reuters.words(documents[i]))
    document_words = re.sub('[^a-zA-Z]', ' ', ' '.join(document_words))
    document_words = document_words.lower()
    document_words = document_words.split()
    ps = PorterStemmer()
    document_words = [ps.stem(word) for word in document_words if not word in set(stopwords.words('english'))]
    document_words = ' '.join(document_words) 
    corpus.append(document_words)
    corpus_tokenize.append(document_words)
    


#conver corpus to list of a list
corpus = ' '.join(corpus) # convert a list to a string
corpus = corpus.split()

id2word = corpora.Dictionary([corpus])

# Create Corpus
texts = [corpus]

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]



# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=5,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis

#X = lda.datasets.load_reuters()
#vocab = lda.datasets.load_reuters_vocab()
#titles = lda.datasets.load_reuters_titles()
#X.shape
#model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)

#model.fit(X)  # model.fit_transform(X) is also available
#topic_word = model.topic_word_  # model.components_ also works
#n_top_words = 8
#for i, topic_dist in enumerate(topic_word):
#    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
#    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


