import math
import numpy as np
from nbc_train import likelihood, build_vocab
from nbc_prior import log_prior
from data_wrangler import normalize_sentence, extract_from_files

def predict_class(d):
    doc = preprocess_test_data(d)
    log_lik = likelihood()
    log_priors = log_prior()
    words = list(build_vocab(extract_from_files()).keys())
    pred_sentence = []
    for sentence in doc:
        sentence = sentence.split(" ")
        pred_class = {}
        for f_class in log_priors:
            sum_c = log_priors[f_class]
            for word in sentence:
                if word in words:
                    sum_c *= log_lik[word][f_class]
            pred_class[f_class] = sum_c
        pred_sentence.append(pred_class)
    
    pred_output = []
    p_c = 0
    for i in range(len(pred_sentence)):
        p_c = np.argmax(list(pred_sentence[i].values()))
        print(p_c)
        p_class = [c for c,f in pred_sentence[i].items() if f==p_c]
        #print(p_class)
        pred_output.append(p_class)
    print(len(pred_sentence))

    pred_output = "\n".join(pred_output)
    return pred_class

def preprocess_test_data(d):
    doc = []
    fo = open(d, "r")
    for line in fo:
        f_split = line.split('\t')
        # Perform sentence normalization on the feature
        sentence = normalize_sentence(f_split[0])
        doc.append(sentence)
    return doc
    
predict_class('dataset/amazon_cells_labelled.txt')


