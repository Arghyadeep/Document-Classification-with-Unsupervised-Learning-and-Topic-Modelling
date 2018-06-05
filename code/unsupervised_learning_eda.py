#dependecies
import pandas as pd
import numpy as np
import pickle as pkl
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics
import matplotlib.ticker as plticker

print("running unsupervised learning on data without labels...")
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



def k_means_silhouette_score(data,n_clusters_range):
    kmeans_silhouette_scores = []
    inertia_scores = {}
    for n_clusters in range(2,n_clusters_range,5):
        clf = KMeans(n_clusters = n_clusters)
        clf.fit(data)
        inertia_scores[n_clusters] = clf.inertia_
        centroids = clf.cluster_centers_
        labels = clf.labels_
        kmeans_silhouette_score = metrics.silhouette_score(data, labels, metric='euclidean')
        kmeans_silhouette_scores.append(kmeans_silhouette_score)
    return kmeans_silhouette_scores,inertia_scores


def generate_inertia_scores(data,n_clusters):
    scores_dict = {}
    inertia = []
    for clusters in n_clusters:
        clf = KMeans(n_clusters = clusters)
        clf.fit(data)
    
       
def plot_silhouette_scores(k_means_silhouette_scores,n_clusters_range,vector_type):
    graph = plt.plot(range(2,n_clusters_range,5),k_means_silhouette_scores)
    plt.title('K-Means')
    plt.xlabel('Cluster')
    plt.ylabel('Silhouette Score')
    plt.savefig("../EDA/kmeans/silhouettes/"+vector_type+".png")
    

def plot_inertia_scores(inertia_dict,nclusters,vector_type):
    for i in range(2,nclusters,5):
        graph = plt.scatter(i,inertia_dict[i])
        plt.title('K-Means')
        plt.xlabel('Cluster')
        plt.ylabel('Inertia Score')
        plt.savefig("../EDA/kmeans/inertias/"+vector_type+".png")


####modularize this code####

#read vectorized data for tfidf vectors
df_tfs_vecs = read_pickle('tfs_vecs.pkl')
df_doc_vecs = read_pickle('doc_vecs.pkl')
df_weighted_doc_vecs = read_pickle('weighted_doc_vecs.pkl')


#plot silhouette scores for tfidf vectors
vectors = np.array(df_doc_vecs.drop(['labels'],1))
result_array1,inertia1 = k_means_silhouette_score(vectors,25)
plot_silhouette_scores(result_array1,25,"tfidf")
plot_inertia_scores(inertia1,25,"tfidf")


#plot silhouette scores for document vectors
vectors = np.array(df_tfs_vecs.drop(['labels'],1))
result_array2,inertia2 = k_means_silhouette_score(vectors,25)
plot_silhouette_scores(result_array2,25,"document_vectors")
plot_inertia_scores(inertia2,25,"document_vectors")

#plot silhouette scores for weighted document vectors
vectors = np.array(df_weighted_doc_vecs.drop(['labels'],1))
result_array3,inertia3 = k_means_silhouette_score(vectors,25)
plot_silhouette_scores(result_array3,25,"weighted_document_vectors")
plot_inertia_scores(inertia3,25,"weighted_document_vectors")

print("best silhouette score for tfidf vectors is", max(result_array1))
print("best silhouette score for document vectors is", max(result_array2))
print("best silhouette score for weighted document vectors is", max(result_array3))

