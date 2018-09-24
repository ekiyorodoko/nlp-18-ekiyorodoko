import math
import numpy as np
import string
import nbc_pred as pred
import nbc_train as train
import data_wrangler as dw


paths = ["dataset/imdb_labelled.txt","dataset/yelp_labelled.txt", "dataset/amazon_cells_labelled.txt" ]
dataset = dw.pre_process(paths)
# Select documents of reviews from dataset
np_dataset = np.array(dataset).T
doc = np_dataset[0]
# Seperate classes from the dataset
classes = np_dataset[1]
unique, counts = np.unique(classes, return_counts = True)
# class categories
C = dict(zip(unique, counts))
# convert corpus to a string of words and then a list of words
BoW = " ".join(doc).split(" ")
# remove empty strings in bag of words
if '' in BoW:
    BoW.remove('')
# Make list of words unique
u_words, count = np.unique(BoW, return_counts = True)
V = dict(zip(u_words, count))

def main():
    nb_train = train.train_nb(C,V, doc, dataset)
    filename = input("Please enter the path to your file:\n")
    nb_test = pred.test(filename, nb_train[0], nb_train[1], C, V)

    fw = open("results_file.txt", "w")
    fw.write("\n".join(nb_test))
    fw.close()

main()