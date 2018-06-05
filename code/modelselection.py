#dependencies
import glob
import pandas as pd
import os
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
from scipy.sparse import *
from scipy import *
import numpy as np
from sklearn.linear_model import LogisticRegression
import operator

print("selecting model...") 

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



#classifiers = ['svm','knn','rf','lr']
#kernel = ['rbf','linear','poly','sigmoid']
#solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
#data = read_pickle("pickle name")
def k_folds_cross_validation(data, kfolds, classifiers, kernel, solvers):
    sv = defaultdict(list)
    knn = defaultdict(list)
    rf = defaultdict(list)
    lr = defaultdict(list)

    #print(data)

    #train and test data generation
    X = np.array(data.drop(['labels'],1))
    y = np.array(list(map(int,data['labels'])))

    #split for cross validation
    split = int(data.shape[0]/kfolds)
    
    for i in range(kfolds):
        
        #index values for the testing set
        x_test = X[int(split*i):int(split + split*i)]
        y_test = y[int(split*i):int(split + split*i)]
        
        #drop test vectors from training matrix
        new_range = [z for z in range(split*i,split+split*i)]
        
        x_train = np.delete(X,new_range,axis=0)
        y_train = np.delete(y,new_range,axis=0)

        
        
        for classifier in classifiers:
            if classifier == 'svm':
                for ker in kernel:
                    sv_classifier = svm.SVC(kernel=ker)
                    sv_classifier.fit(x_train, y_train)
                    temp = sv_classifier.score(x_test,y_test)
                    sv[classifier+"_"+ker].append(temp)

                   
            #play with upper range
            if classifier == 'knn':
                for n in range(5,50,5):
                    knc = KNeighborsClassifier(n_neighbors=n)
                    knc.fit(x_train, y_train)
                    temp = knc.score(x_test,y_test)
                    knn[classifier+"_"+str(n)].append(temp)
                    
            #play with upper range   
            if classifier == 'rf':
                for n in range(5,50,5):
                    rfc = RandomForestClassifier(n_estimators=n)
                    rfc.fit(x_train, y_train)
                    temp = rfc.score(x_test,y_test)
                    rf[classifier+"_"+str(n)].append(temp)


            if classifier == 'lr':
                for sol in solvers:
                    log_reg = LogisticRegression(solver = sol)
                    log_reg.fit(x_train, y_train)
                    temp = log_reg.score(x_test,y_test)
                    lr[classifier+"_"+sol].append(temp)

    
    scores = [dict(sv),dict(knn),dict(rf),dict(lr)]
    
    average_scores = {}
    for model in scores:
        for params in model:
            average_scores[params] = sum(model[params])/float(10)


    return average_scores

def model_selection(average_scores,preprocessing_type):
    sorted_dict = sorted(average_scores.items(), key=operator.itemgetter(1), reverse=True)
    
    ####modularize with dump_pickle#### temporary pickling for EDA ########################
    pkl.dump(average_scores, open("../pickles/modelselection_"+preprocessing_type+".pkl","wb"))
    ########################################################################################
    
    selected_model = str(sorted_dict[0][0])
    top_average_accuracy = str(sorted_dict[0][1]*100) + "%"
    print("Top 10 models for "+ preprocessing_type + " on 10 fold cross validation are: ")
    for model,accuracy in (sorted_dict[:10]):
        print(str(model)+ " with accuracy " +str(accuracy*100)+"%")
    print("selected model for "+ preprocessing_type + " is " + selected_model +
          " with average accuracy on 10 folds cross validation : " + top_average_accuracy)
    return selected_model.split("_")


######need to create a loop or function for the following code#######

#getting data for doc_vecs, tfidf vectors and weighted tf idf vectors    
data_doc_vecs = read_pickle("doc_vecs.pkl")
data_tfidf_vecs = read_pickle("tfs_vecs.pkl")
data_weighted_doc_vecs = read_pickle("weighted_doc_vecs.pkl")


      
#model selction for document vectors
model_doc_vecs,hyperparameter_doc_vecs = model_selection(k_folds_cross_validation(data_doc_vecs, 10,
                                 ['svm','knn','rf','lr'],
                                 ['rbf','linear','poly','sigmoid'],
                                 ['newton-cg', 'lbfgs', 'liblinear']),"document vectors")

#model selction for tf idf vectors
model_tfidf_vecs,hyperparameter_tfidf_vecs = model_selection(k_folds_cross_validation(data_tfidf_vecs, 10,
                                 ['svm','knn','rf','lr'],
                                 ['rbf','linear','poly','sigmoid'],
                                 ['newton-cg', 'lbfgs', 'liblinear']),"tfidf vectors")

#model selction for weighted document vectors
model_weighted_doc_vecs,hyperparameter_weighted_doc_vecs = model_selection(k_folds_cross_validation(data_weighted_doc_vecs, 10,
                                 ['svm','knn','rf','lr'],
                                 ['rbf','linear','poly','sigmoid'],
                                ['newton-cg', 'lbfgs', 'liblinear']),"tfidf weighted document vectors")


##########need to modify dump_pickle ############3 modularize code

#dumping document vector model
pkl.dump(model_doc_vecs,open("../pickles/best_model_doc_vecs.pkl","wb"))
pkl.dump(hyperparameter_doc_vecs,open("../pickles/hyperparameter_doc_vecs.pkl","wb"))

#dumping tf idf vectors model
pkl.dump(model_tfidf_vecs,open("../pickles/best_model_tfidf_vecs.pkl","wb"))
pkl.dump(hyperparameter_tfidf_vecs,open("../pickles/hyperparameter_tfidf_vecs.pkl","wb"))

#dumping weighted document vectors model
pkl.dump(model_weighted_doc_vecs,open("../pickles/best_model_weighted_doc_vecs.pkl","wb"))
pkl.dump(hyperparameter_weighted_doc_vecs,open("../pickles/hyperparameter_weighted_doc_vecs.pkl","wb"))


###################################################################
    



    

