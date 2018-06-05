# dependencies
import pandas as pd
import pickle as pkl

print("Splitting data to train and test files...")
# shuffle data
def shuffle(df):
    return df.sample(frac = 1).reset_index(drop = True)

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


def train_test_split(df, split = 0.8):
    df = shuffle(df)
    df_train = df.iloc[:int(split*len(df))]
    df_test = df.iloc[int(split*len(df)):]
    dump_pickle(df_train, 'cleaned_train_data.pkl')
    dump_pickle(df_test, 'cleaned_test_data.pkl')


df = read_pickle('cleaned_data.pkl')
train_test_split(df)

train = read_pickle("cleaned_train_data.pkl")
#print(train.head())
#print(train['labels']) #wtf!!!
test = read_pickle("cleaned_test_data.pkl")
#print(test.head()) #works fine
#print(test["labels"]) #need to fix this thing
    
