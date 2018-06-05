#dependencies
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
import pickle as pkl


####code needs to be modularized and formatted properly#####

print("running cleaned EDA...")
def read_pickle(pickle_name):
    path = "../pickles/"
    return pkl.load(open(path+pickle_name, "rb"))

label = 'labels'
texts = 'texts'

data = read_pickle("cleaned_data.pkl")


doc_count = {}
for i in range(len(data)):
    if data[label][i] in doc_count:
        doc_count[data[label][i]] += 1
    else:
        doc_count[data[label][i]] = 1



    
total_word_set = {}
for docs in data[texts]:
    for word in docs:
        if word in total_word_set:
            total_word_set[word] += 1
        else:
            total_word_set[word] = 1


sorted_dict = sorted(total_word_set.items(), key=operator.itemgetter(1), reverse=True)
top_freqs = []
top_words = []
for i in (sorted_dict[:20]):
    top_freqs.append(i[1])
    top_words.append(i[0])

top_words = np.array(top_words)
top_freqs = np.array(top_freqs)
plt.bar(range(len(top_freqs)),top_freqs,align='center' )
plt.xticks(range(len(top_words)),top_words,rotation=90)
plt.tight_layout()
#plt.show()
plt.savefig("../EDA/cleaned/word_frequency_global.png")



labels = list(set(data[label]))
dict_by_categories = {}

for i in labels:
    total_word_set_2 = {}
    for j in range(len(data)):
        doc = data[texts][j]
        if data[label][j] == i:
            for word in doc:
                if word in total_word_set_2:
                    total_word_set_2[word] += 1
                else:
                    total_word_set_2[word] = 1
    dict_by_categories[i] = total_word_set_2


sorted_dict_by_categories = {}
for i in dict_by_categories:
    sorted_dict = sorted(dict_by_categories[i].items(), key=operator.itemgetter(1), reverse=True)
    sorted_dict_by_categories[i] = sorted_dict


x = 1
for i in sorted_dict_by_categories:

    top_words_pairs2 = sorted_dict_by_categories[i][:20]
    top_words = []
    top_freqs = []
    top_freqs_global = []
    for j in top_words_pairs2:
        top_words.append(j[0])
        top_freqs.append(int(j[1])/doc_count[i])
        top_freqs_global.append(int((total_word_set[j[0]]))/len(data))

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.6, 0.9, 0.5],ylim = (0,max(top_freqs)+1))
    ax2 = fig.add_axes([0.1, 0.1, 0.9, 0.5],ylim = (0,max(top_freqs_global)+1))
    ax2.invert_yaxis()
    ax1.bar(range(len(top_freqs)),top_freqs, color = "blue")
    ax2.bar(range(len(top_freqs_global)),top_freqs_global, color = 'green')
    plt.xticks(np.arange(len(top_words)),top_words, rotation=90)
    plt.savefig("../EDA/cleaned/"+"label"+str(x)+".png")
    x += 1
    


##raw_data = pd.read_csv("../data/labeled_data.csv")
##
##data = raw_data
##label = '7CAT'
##texts = 'original_post'
##
##doc_count = {}
##for i in range(len(data)):
##    if data[label][i] in doc_count:
##        doc_count[data[label][i]] += 1
##    else:
##        doc_count[data[label][i]] = 1
##
##    
##total_word_set = {}
##for docs in data[texts]:
##    for word in docs.split():
##        if word in total_word_set:
##            total_word_set[word] += 1
##        else:
##            total_word_set[word] = 1
##
##
##sorted_dict = sorted(total_word_set.items(), key=operator.itemgetter(1), reverse=True)
##top_freqs = []
##top_words = []
##for i in (sorted_dict[:20]):
##    top_freqs.append(i[1])
##    top_words.append(i[0])
##
##top_words = np.array(top_words)
##top_freqs = np.array(top_freqs)
##plt.bar(range(len(top_freqs)),top_freqs,align='center' )
##plt.xticks(range(len(top_words)),top_words,rotation=90)
##plt.tight_layout()
###plt.show()
##plt.savefig("../EDA/cleaned/word_frequency_global_raw.png")   
##                
##
##
##
##

