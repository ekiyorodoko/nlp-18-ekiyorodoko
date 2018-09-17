import numpy as np
import math
import string

# INPUT
#   d   :   a list of path to files

# OUTPUT
#   f   :   a 2-d array of the stringed feature
#   V   :   an array of unique words from the document 

def main(d):
    extract_from_files(d)


# Extract data from files
def extract_from_files(d):

    # Initialize feature list
    features = []

    # Loop over provided list to read each documents
    for i in range(len(d)):
        fo = open(d[i], "r")

    # Get lines from document and split it into feature(string) 
    # and class(string) then append to the features array.
        for line in fo:
            f_split = line.split('\t')
            feature = f_split[0]
            f_class = str(int(f_split[1]))
            features.append([feature, f_class])
            
        fo.close()

    # return an ndarray convert to numpy array
    return np.array(features)

# Normalize the sentences: Focus on key aspects of the sentences to 
# get better prediction results.
def normalize_sentence(feat):
    # Step 1: Extract words(group of character's before a space including 
    # punctuations) from sentences
    words = feat.split(" ")
    # Step 2: Remove punctuations from words
    nwords = words.translate(str.maketrans('','',string.punctuation))
    # Step 3: Convert words to lower case
    norm_feature = nwords.lower()  

    return norm_feature



#def build_vocab(words):



