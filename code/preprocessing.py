# dependences
import pandas as pd
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import numpy as np
from gensim import models
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

print("preprocessing...")
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

# returns mapping of labels to integers
def label_map(labels):
    label_dict = {}
    temp1 = list(set(labels))
    temp2 = [(i+1) for i in range(len(temp1))]
    for i in range(len(temp1)):
        label_dict[temp1[i]] = int(temp2[i])
    mapped_labels = []
    for label in labels:
        mapped_labels.append(label_dict[label])
    return mapped_labels, label_dict# computes tfs matrix for texts

# dumps dataframe
# returns dataframe
def tf_idf(texts):
    tfidf = TfidfVectorizer()
    tfs = tfidf.fit_transform(texts)
    tfs_df = pd.DataFrame(tfs.A, columns = tfidf.get_feature_names())
    vocab = tfidf.get_feature_names()
    return tfs_df, vocab

# creates one document string from tokenized document
# returns list of strings
def untokenize(texts):
    docs = []
    for doc in texts:
        temp = ""
        for word in doc:
            temp += word + " "
        docs.append(temp)
    return docs

# creates dictionary with word:vector key value pairs
# dumps dataframe
# returns dict
def w2v(texts, size):
    model = Word2Vec(texts, size = size)
    word_dict = {}
    for word in model.wv.index2word:
        word_dict[word] = model[word]
##    make this syntax work
##    word_dict = dict(zip(model.wv.index2word, model.wv.vectors))
    return word_dict

# doc vecs computed as simple average of word vectors within the doc
# texts is DataFrame, word_dict is dict, size is int
# dumps doc vecs as data_frame
# returns doc vecs as list
def d2v(texts, word_dict, size):
    doc_vecs = []
    for doc in texts:
        counter = 0
        vecs = np.zeros(size)
        for words in doc:
            if words in word_dict:
                vecs = np.add(vecs,word_dict[words])
                counter += 1
        vecs = np.divide(vecs,counter)
        doc_vecs.append(vecs)
    return doc_vecs

# texts = list of documents
# tfs_df = dataframe of tfidf values
# word_dict = dict that maps words to w2v vectors
# vocab = vocabulary of tfidf
# size = size of w2v vectors
def tfidf_d2v(texts, tfs_df, word_dict, vocab, size):
    doc_vecs = []
    for i in range(len(tfs_df)):
        n = 0
        vectors = [0]*size
        for word in vocab:
            if word in word_dict:
                vectors += word_dict[word]*tfs_df[word][i]
                n += 1
        doc_vecs.append([i/n for i in vectors])
    return doc_vecs

def tsne(vecs, perplexity = 30):
    model_tsne = TSNE(n_components=2, verbose = 1, perplexity = perplexity,
                  learning_rate = 30, random_state=0)
    pca = PCA(n_components = 2, svd_solver='full')
    #feed pca vecs to tsne
    tsne = model_tsne.fit_transform(vecs)
    return tsne

def create_df(vectors, labels, name):
    """
    creates dataframe of vectors paired with associated labels
    and dumps a pickle of the dataframe.
    vectors is list of lists.
    labels is list.
    name is string.
    """
    cols = ["x"+str(i) for i in range(len(vectors[1]))]
    df = pd.DataFrame(vectors, columns = cols)
    df["labels"] = labels
    dump_pickle(df, name+".pkl")



#######need to organize this part######
labels = read_pickle("cleaned_train_data.pkl")['labels']
texts = read_pickle("cleaned_train_data.pkl")['texts']
size = 50

test_labels = read_pickle("cleaned_test_data.pkl")['labels']
test_texts = pkl.load(open("../pickles/cleaned_test_data.pkl","rb"))['texts']
test = read_pickle("cleaned_test_data.pkl")
#print(test)
#print(list(test_texts))


tfs_df, vocab = tf_idf(untokenize(texts))
dic = w2v(texts, size)
doc_vecs = d2v(texts, dic, size)
weighted_doc_vecs = tfidf_d2v(texts, tfs_df, dic, vocab, size)

#creating train dataframes
create_df(tfs_df.values.tolist(), labels, "tfs_vecs")
create_df(doc_vecs, labels, "doc_vecs")
create_df(weighted_doc_vecs, labels, "weighted_doc_vecs")

#creating vectors
test_tfs_df, test_vocab = tf_idf(untokenize(test_texts))
test_dic = w2v(test_texts, size)
test_doc_vecs = d2v(test_texts, test_dic, size)
test_weighted_doc_vecs = tfidf_d2v(test_texts, test_tfs_df, test_dic, test_vocab, size)

#tsne functions
tfidf_tsne = tsne(tfs_df)
d2v_tsne = tsne(doc_vecs)
weighted_d2v_tsne = tsne(weighted_doc_vecs)

#create test dataframes (need to modify test data with vocab)
create_df(test_tfs_df.values.tolist(), test_labels, "test_tfs_vecs")
create_df(test_doc_vecs, test_labels, "test_doc_vecs")
create_df(test_weighted_doc_vecs, test_labels, "test_weighted_doc_vecs")

# create tsne dataframes
create_df(tfidf_tsne, labels, "tfidf_tsne")
create_df(d2v_tsne, labels, "d2v_tsne")
create_df(weighted_d2v_tsne, labels, "weighted_d2v_tsne")

