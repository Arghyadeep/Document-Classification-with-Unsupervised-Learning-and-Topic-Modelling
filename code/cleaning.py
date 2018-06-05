# dependencies 
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
import pickle as pkl
import os
import sys
from nltk.stem import WordNetLemmatizer

print("cleaning data...")
# read in .csv
# needs exception for non-existant file #exception needs to be modified!
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

# load pickle
# pickle must be of type DataFrame
def read_pickle(pickle_name):
    path = "../pickles/"
    obj = pkl.load(open(path+pickle_name, "rb"))
    if not type(obj) == type(pd.DataFrame()):
        raise TypeError("object to read must be DataFrame")
    return obj

# dumps a pickle of obj
# only DataFrames permitted
def dump_pickle(obj, pickle_name):
    path = "../pickles/"
    if not type(obj) == type(pd.DataFrame()):
        raise TypeError("object to dump must be DataFrame")
    pkl.dump(obj, open(path+pickle_name, "wb"))

# converts to lowercase, removes punctuation, tokenize
def pre_clean(texts):
    texts = [word.lower() for word in texts]
    tokens = [word_tokenize(doc) for doc in texts]
    #lemmatizer
    l = WordNetLemmatizer()
    lemmatized_docs = []
    for token in tokens:
        temp = []
        for word in token:
            temp.append(l.lemmatize(word))
        lemmatized_docs.append(temp)
    return lemmatized_docs

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

def drop_filter(texts, stop_words = list(stopwords.words('english')), custom_sw = ["ip","crypto","vpn","isakmp","ipsec","asa","cisco","hi","thanks","one","lt","anyconnect"]):
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

def create_df(texts, labels):
    columns = ["texts", "labels"]
    index = range(len(texts))
    df = pd.DataFrame(columns = columns, index = index)
    df["texts"] = texts
    df["labels"] = labels
    dump_pickle(df, 'cleaned_data.pkl')


filename = "labeled_data.csv"#input("enter .csv: ")
texts = "original_post"#input("enter text field name: ")
label = "7CAT"#input("enter label field name: ")
texts, labels = read_file(filename, texts, label)
    

# missing logic for handling drop_set vs keep_set
filtered_texts = pre_clean(texts)
filtered_texts = number_filter(filtered_texts)
filtered_texts = drop_filter(filtered_texts)
    
create_df(filtered_texts,labels)
    
