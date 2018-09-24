import numpy as np
import math
import string

paths = ["dataset/imdb_labelled.txt","dataset/yelp_labelled.txt", "dataset/amazon_cells_labelled.txt" ]

def pre_process(d):
    dataset = []
    # Loop over provided list to read each documents
    for i in range(len(d)):
        fo = open(d[i], "r")

    # Get lines from document and split it into feature(string) 
    # and class(string) then append to the features array.
        for line in fo:
            f_split = line.split('\t')
            
            # Perform sentence normalization on the feature
            
            feature = f_split[0]
            f_class = str(int(f_split[1]))

            # Step 1: Remove punctuations from words
            nwords = feature.translate(str.maketrans('','',string.punctuation))
            # Step 2: Convert words to lower case
            norm_feature = nwords.lower()


            #Add to features array
            dataset.append([norm_feature, f_class])
        fo.close()
    return dataset
