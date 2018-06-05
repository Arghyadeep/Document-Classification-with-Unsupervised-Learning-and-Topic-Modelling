#dependencies
import glob
import os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
from scipy.sparse import *
from scipy import *
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd


print("testing")
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


##def select_best_classifier_and_hyperparameter(classifiers):
##    pass
##    #return best classifier with best hyperparameter in a saved variable as string



def test(test_data,model):
    X = np.array(test_data.drop(['labels'],1))
    y = np.array(list(map(int,test_data['labels'])))
    accuracy = model.score(X,y)
    return accuracy

######need to create a loop or function for the following code#######

#Testing for document vectors
model_doc_vecs = pkl.load(open("../pickles/trained_model_doc_vecs.pkl","rb"))
data_doc_vecs = read_pickle("test_doc_vecs.pkl")

temp = read_pickle("cleaned_test_data.pkl")

data_doc_vecs['labels'] = list(temp['labels']) #why the fuck do I have to do this!!!
print("test accuracy is " + str(test(data_doc_vecs,model_doc_vecs)*100) + "%" )

#Testing for tfidf vectors
model_tfidf_vecs = pkl.load(open("../pickles/trained_model_tfidf_vecs.pkl","rb"))
data_tfidf_vecs = read_pickle("test_tfs_vecs.pkl")

temp = read_pickle("cleaned_test_data.pkl")

data_tfidf_vecs['labels'] = list(temp['labels']) #why the fuck do I have to do this!!!
#print("test accuracy is on document vectors is " + str(test(data_tfidf_vecs,model_tfidf_vecs)*100) + "%" )

#testing for weighted document vectors
model_weighted_doc_vecs = pkl.load(open("../pickles/trained_model_weighted_doc_vecs.pkl","rb"))
data_weighted_doc_vecs = read_pickle("test_weighted_doc_vecs.pkl")

temp = read_pickle("cleaned_test_data.pkl")

data_weighted_doc_vecs['labels'] = list(temp['labels']) #why the fuck do I have to do this!!!
#print("test accuracy is on weighted document vectors is " + str(test(data_weighted_doc_vecs,model_weighted_doc_vecs)*100) + "%" )




#make this code general for any preprocessing
#accuracy = test(data,model)
#print("accuracy on test data ",accuracy)

