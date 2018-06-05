#dependencies
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import matplotlib.cm as cm
import numpy as np


print("running model selection EDA...")

def read_pickle(pickle_name):
    path = "../pickles/"
    return pkl.load(open(path+pickle_name, "rb"), encoding = 'latin1')

def plot_model_accuracies(data,vector_type):
    models = []
    accuracies = []
    for i in data:
        models.append(i)
        accuracies.append(data[i])

    models = np.array(models)
    accuracies = np.array(accuracies)

    plt.bar(range(len(accuracies)),accuracies, align = 'center')
    plt.xticks(range(len(models)),models, rotation = 90)
    plt.savefig("../EDA/modelselection/"+vector_type+".png")

data_tfidf = read_pickle("modelselection_tfidf vectors.pkl")
data_doc_vecs = read_pickle("modelselection_document vectors.pkl")
data_weighted_doc_vecs = read_pickle("modelselection_tfidf weighted document vectors.pkl")


plot_model_accuracies(data_tfidf,"tfidf")
plot_model_accuracies(data_doc_vecs,"doc_vecs")
plot_model_accuracies(data_weighted_doc_vecs,"weighted_doc_vecs")
