#dependencies
import glob
import os
import sklearn
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
from scipy.sparse import *
from scipy import *
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys


print("training on data...")



def read_pickle(pickle_name):
    path = "../pickles/"
    return pkl.load(open(path+pickle_name, "rb"))

def dump_pickle(obj, pickle_name):
    path = "../pickles/"
    pkl.dump(obj, open(path+pickle_name, "wb"))

def select_best_classifier_and_hyperparameter(classifier,hyperparameter):
    classifier_dict = {"svm":"svm.SVC","knn":"KNeighborsClassifier",
                       "rf":"RandomForestClassifier","lr":"LogisticRegression"}

    parameter_names = {"svm":"kernel","knn":"n_neighbours",
                       "rf":"n_estimators","lr":"solver"}

    object_conversions = {"svm":sklearn.svm.classes.SVC,
                          "knn":sklearn.neighbors.classification.KNeighborsClassifier ,
                          "lr": sklearn.linear_model.logistic.LogisticRegression ,
                          "rf":sklearn.ensemble.forest.RandomForestClassifier }
    
    classifier_with_hyperparameter = classifier_dict[classifier] + "(" + parameter_names[classifier] + "=" + str(hyperparameter) + ")"
    
    return object_conversions[classifier](classifier_with_hyperparameter)

    
def train(train_data_features, train_data_labels, classifier):
    model = LogisticRegression().fit(train_data_features,train_data_labels)
    return model


######need to create a loop or function for the following code#######

#getting data for doc_vecs, tfidf vectors and weighted tf idf vectors    
data_doc_vecs = read_pickle("doc_vecs.pkl")
data_tfidf_vecs = read_pickle("tfs_vecs.pkl")
data_weighted_doc_vecs = read_pickle("weighted_doc_vecs.pkl")

#creating train features and train labels for doc vecs
X_doc_vecs = np.array(data_doc_vecs.drop(['labels'],1))
y_doc_vecs = np.array(list(map(int,data_doc_vecs['labels'])))

#creating train features and train labels for doc vecs
X_tfidf_vecs = np.array(data_tfidf_vecs.drop(['labels'],1))
y_tfidf_vecs = np.array(list(map(int,data_tfidf_vecs['labels'])))

#creating train features and train labels for doc vecs
X_weighted_doc_vecs = np.array(data_weighted_doc_vecs.drop(['labels'],1))
y_weighted_doc_vecs = np.array(list(map(int,data_weighted_doc_vecs['labels'])))


#classifier for doc vecs
classifier_doc_vecs = select_best_classifier_and_hyperparameter(read_pickle("best_model_doc_vecs.pkl"),
                                                       read_pickle("hyperparameter_doc_vecs.pkl"))

#classifier for tfidf_vecs
classifier_tfidf_vecs = select_best_classifier_and_hyperparameter(read_pickle("best_model_tfidf_vecs.pkl"),
                                                       read_pickle("hyperparameter_tfidf_vecs.pkl"))

#classifier for weighted doc vecs
classifier_weighted_doc_vecs = select_best_classifier_and_hyperparameter(read_pickle("best_model_weighted_doc_vecs.pkl"),
                                                       read_pickle("hyperparameter_weighted_doc_vecs.pkl"))

#model for doc_vecs
model_doc_vecs = train(X_doc_vecs,y_doc_vecs,classifier_doc_vecs)
dump_pickle(model_doc_vecs,"trained_model_doc_vecs.pkl")

#model for tfidf_vecs
model_tfidf_vecs = train(X_tfidf_vecs,y_tfidf_vecs,classifier_tfidf_vecs)
dump_pickle(model_tfidf_vecs,"trained_model_tfidf_vecs.pkl")

#model for weighted doc_vecs
model_weighted_doc_vecs = train(X_weighted_doc_vecs,y_weighted_doc_vecs,classifier_weighted_doc_vecs)
dump_pickle(model_weighted_doc_vecs,"trained_model_weighted_doc_vecs.pkl")

print("finished training")


