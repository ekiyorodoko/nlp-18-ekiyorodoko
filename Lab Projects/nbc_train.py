import math
import numpy as np
from data_wrangler import extract_from_files
from nbc_prior import log_prior
import re

def build_vocab(f_list):
    # Constraint: Words must be unique
    # Transpose numpy array to extract just the features from the dataset
    #
    corpus = f_list.T[0]
    # convert corpus to a string of words
    concat_corpus = " ".join(corpus)
    # convert converted string of words to list of words
    split_corpus = concat_corpus.split(" ")
    # Make list of words unique
    unique, counts = np.unique(split_corpus, return_counts = True)
    bag_of_words = dict(zip(unique, counts))

    return bag_of_words

def likelihood():
    features = extract_from_files()
    log_priors = log_prior()
    V = build_vocab(features)
    lik = {}
    f_classes = list(log_priors.keys())
    for word in V:
        lik[word] = {}
        for f_class in f_classes:
            vwc = 0
            wc = []
            for feature in features:
                # Check if the feature belongs to the class being iterated
                if feature[1] == f_class:
                    #count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), feature[0]))
                    f_split = feature[0].split(" ")
                    wc.append(f_split.count(word))
                    vwc += len(f_split)

            for i in range(len(wc)):
                pwc = (wc[i] + 1)/(vwc + len(V))
                lik[word][f_class] = pwc

    return lik