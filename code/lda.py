# dependencies 
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
import pickle as pkl
import os
import sys
from sklearn.decomposition import LatentDirichletAllocation as lda, NMF
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import LdaModel

#add parameters
##clf = lda(n_components = 7)
##
##def read_pickle(pickle_name):
##    path = "../pickles/"
##    obj = pkl.load(open(path+pickle_name, "rb"))
##    if not type(obj) == type(pd.DataFrame()):
##        raise TypeError("object to read must be DataFrame")
##    return obj
##
##data = read_pickle("tfs_vecs.pkl")
##X = data.drop(['labels'],1)
##
##model = clf.fit(X)
##for i in (model.components_):
##    print(max(i))


def read_file(filename, texts, labels, encoding = "utf-8"):
    path = "../data/"
    try:
        data = pd.read_csv(path+filename, encoding = encoding)
        docs = data[texts]
        labels = data[labels]
        return docs, labels
    except:
        print ("File does not exist")
        sys.exit()

def tf_idf(texts):
    tfidf = TfidfVectorizer(max_df = 0.95, min_df = 2,
                            max_features = 1000, stop_words = 'english')
    tfs = tfidf.fit_transform(texts)
    tfs_df = pd.DataFrame(tfs.A, columns = tfidf.get_feature_names())
    vocab = tfidf.get_feature_names()
    return tfs_df, vocab

def tf(texts):
    tf = CountVectorizer(max_df = 0.95, min_df = 2,
                            max_features = 1000, stop_words = 'english')
    tfs = tf.fit_transform(texts)
    tfs_df = pd.DataFrame(tfs.A, columns = tf.get_feature_names())
    vocab = tf.get_feature_names()
    return tfs_df, vocab


# converts to lowercase, removes punctuation, tokenize
def pre_clean(texts):
    texts = [doc.lower() for doc in texts]
    tokens = [word_tokenize(doc) for doc in texts]
    return tokens

# filters out numbers
# text parameter is tokenized docs
def number_filter(texts, custom_dict = []):
    filter_set = set(custom_dict)
    filtered_docs = []
    for doc in texts:
        filtered_doc = []
        for word in doc:
            if word in filter_set or word.isalpha():
                filtered_doc.append(word)
        filtered_docs.append(filtered_doc)
    return filtered_docs

def drop_filter(texts, stop_words = list(stopwords.words('english')), custom_sw = []):
    stop_words = set(stop_words).union(set(custom_sw))
    filtered_docs = []
    for doc in texts:
        filtered_doc = []
        for word in doc:
            if not word in stop_words:
                filtered_doc.append(word)
        filtered_docs.append(filtered_doc)
    return filtered_docs

def keep_filter(texts, std_words = list(words.words()), custom_words = [], std_sw = list(stopwords.words('english')), custom_sw = []):
    words = set(std_words).union(set(custom_words))
    stop_words = set(std_sw).union(set(custom_sw))
    for doc in texts:
        filtered_doc = []
        for word in doc:
            if word in words and not word in stop_words:
                filtered_doc.append(word)
        filtered_docs.append(filtered_doc)
    return filtered_docs

def untokenize(texts):
    docs = []
    for doc in texts:
        temp = ""
        for word in doc:
            temp += word + " "
        docs.append(temp)
    return docs

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic %d: " % topic_idx
        message += ",".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-3]])
        print(message)
    print()
    print()

filename = "labeled_data.csv"#input("enter .csv: ")
texts = "original_post"#input("enter text field name: ")
label = "5CAT"#input("enter label field name: ")
texts, labels = read_file(filename, texts, label)

filtered_texts = pre_clean(texts)
filtered_texts = number_filter(filtered_texts)
filtered_texts = drop_filter(filtered_texts)
texts = untokenize(filtered_texts)
#tf,voc = tf_idf(filtered_texts)
#print(tf.head(2))

tf,voc = tf(texts)

clf = lda(n_components = 8)
model = clf.fit(tf)


##print(model.components_)
##print(len(model.components_[0]))
##for i in (model.components_):
##    print(max(i))
##print(len(model.components_))

print_top_words(model, voc, 20)



#trans = model.transform(texts)

#print(trans)



