import math
import numpy
from nbc_train import likelihood
from nbc_prior import log_prior
from data_wrangler import normalize_sentence, extract_from_files

def predict_class(d):
    doc = preprocess_test_data(d)
    log_lik = likelihood()
    log_priors = log_prior()
    words = list(extract_from_files().keys())
    pred_class = []
    for f_class in log_priors:
        sum_c = log_priors[f_class]
        pred_sentence = []
        for sentence in doc:
            sentence = sentence.split(" ")
            for word in sentence:
                if word in words:
                    sum_c += log_lik[word][f_class]
            pred_sentence.append(sum_c)
        pred_class.append(pred_sentence)
    
    
        
            

            





def preprocess_test_data(d=['dataset/yelp_labelled.txt']):
    doc = []
    fo = open(d, "r")
    for line in fo:
        f_split = line.split('\t')
        # Perform sentence normalization on the feature
        sentence = normalize_sentence(f_split[0])
        doc.append(sentence)
    return doc
    



