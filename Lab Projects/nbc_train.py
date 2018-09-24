import numpy as np
import math


def train_nb(C,V, doc, dataset):
    Ndoc = len(doc)
    log_priors = {}
    log_liks = {}
    for c in C:
        log_liks[c] = {}
        Nc = C[c]
        log_prior = math.log(Nc/Ndoc)
        log_priors[c] = log_prior
        arb_array = []
        for data in dataset:
            if (data[1]==c):
                arb_array.append(data[0])
        #bigdoc[c]=arb_array

        c_BoW = " ".join(arb_array).split(" ")
        for w in V:
            count_wc = c_BoW.count(w)
            log_lik = math.log((count_wc + 1)/len(c_BoW))
            log_liks[c][w] = log_lik

    return log_priors,log_liks