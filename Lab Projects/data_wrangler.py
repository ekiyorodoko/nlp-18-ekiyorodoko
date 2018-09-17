import numpy as np
import math

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
    


