import numpy as np
import math
import string
import data_wrangler as dw

# Build a vocabulary of words from the feature sentences


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
        new_class_dict[key] = Nc/Ndoc
    return new_class_dict


