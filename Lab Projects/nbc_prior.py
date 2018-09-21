import numpy as np
import math
import string
import data_wrangler as dw

# Build a vocabulary of words from the feature sentences

def build_vocab(f_list):
    bag_of_words = []
    # Constraint: Words must be unique
    # Transpose numpy array to extract just the features from the dataset
    #
    corpus = f_list.T[0]
    # convert corpus to a string of words
    concat_corpus = " ".join(corpus)
    # convet converted string of words to list of words
    split_corpus = concat_corpus.split(" ")
    # Make list of words unique


    return bag_of_words

def class_count():
    features = dw.extract_from_files()
    classes = features.T[1]
    unique, counts = np.unique(classes, return_counts = True)
    class_dict = dict(zip(unique, counts))
    return class_dict

def log_prior():
    new_class_dict = class_count()
    Ndoc = 0
    for c in new_class_dict:
        Ndoc += new_class_dict[c]
    for key in new_class_dict:
        Nc = new_class_dict[c]
        new_class_dict[key] = log(Nc/Ndoc)
    return new_class_dict

print(log_prior())

