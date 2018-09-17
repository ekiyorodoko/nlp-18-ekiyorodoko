import numpy as np
import math
import string

# INPUT
#   d   :   a list of path to files

# OUTPUT
#   dataset  :   a 2-d numpy array of normalized feature and it's class.
#   bag_of_words   :   an array of unique words from the document 

def main(d):
    dataset = extract_from_files(d)


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
            # Perform sentence normalization on the feature
            feature = normalize_sentence(f_split[0])
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




