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

def priors():
    features = dw.extract_from_files()
    classes = features[1]
    unique, counts = np.unique(classes, return_count = True)
    prior_dict = dict(zip(unique, counts))
    return prior_dict

print(priors)

