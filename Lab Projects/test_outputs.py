import  numpy as np
import  math
from data_wrangler import extract_from_files

# Tests the correctnes of every file output in this
# project.

def main():
    paths = ["dataset/yelp_labelled.txt"]
    print(feature_test(paths))

#   Specific test for feature output from data_wrangler.py
def feature_test(d):
    f = extract_from_files(d)
    shape = f.shape
    return shape

main()