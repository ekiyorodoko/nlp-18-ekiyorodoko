import numpy as np
import math
import data_wrangler as dw

# Tests the correctnes of every file output in this
# project.

def main():
    paths = ["dataset/yelp_labelled.txt"]

    # Output size test result
    if (feature_test(paths)==(3000,2)):
        print("Features passed output size test")
    else:
        print("Features failed output size test")

#   Specific test for feature output
def feature_test(d):
    f = dw.extract_from_files(d)
    shape = f.shape
    return shape

main()