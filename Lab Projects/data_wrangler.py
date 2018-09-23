import numpy as np
import math
import string

# INPUT
#   d   :   a list of path to files

# OUTPUT
#   dataset  :   a 2-d numpy array of normalized feature and it's class.
#   bag_of_words   :   an array of unique words from the document 


# Extract data from files
def extract_from_files():
    d = ['dataset/amazon_cells_labelled.txt']
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
    # Step 1: Remove punctuations from words
    nwords = feat.translate(str.maketrans('','',string.punctuation))
    # Step 2: Convert words to lower case
    norm_feature = nwords.lower()
    # Step 3: Extract words(group of character's before a space including 
    # punctuations) from sentences
    words = norm_feature.split(" ")
    words = " ".join(words)
     
    return words

extract_from_files()



