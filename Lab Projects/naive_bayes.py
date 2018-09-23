import math
import numpy as np
import string
import random

d = ["dataset/imdb_labelled.txt","dataset/yelp_labelled.txt", "dataset/amazon_cells_labelled.txt" ]
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


# Select documents of reviews from dataset
np_dataset = np.array(dataset).T
doc = np_dataset[0]

# Seperate classes from the dataset
classes = np_dataset[1]
unique, counts = np.unique(classes, return_counts = True)

# unique classes
f_c = dict(zip(unique, counts))


# convert corpus to a string of words
BoW = " ".join(doc)
# convert converted string of words to list of words
split_BoW = BoW.split(" ")

if '' in split_BoW:
    split_BoW.remove('')

# Make list of words unique
u_words, count = np.unique(split_BoW, return_counts = True)
V = dict(zip(u_words, count))

def train_nb(D, C):
    Ndoc = len(D)
    log_priors = {}
    log_liks = {}
    for c in C:
        log_liks[c] = {}
        Nc = C[c]
        log_prior = math.log(Nc/Ndoc)
        log_priors[c] = log_prior
        arb_array = []
        for data in dataset:
            if (data[1]==c):
                arb_array.append(data[0])
        #bigdoc[c]=arb_array

        c_BoW = " ".join(arb_array).split(" ")
        for w in V:
            count_wc = c_BoW.count(w)
            log_lik = math.log((count_wc + 1)/len(c_BoW))
            log_liks[c][w] = log_lik

    return log_priors,log_liks

def test(testdoc, logprior, loglikelihood, C, V):
    open_doc = open(testdoc, "r")
    testset = []
    for review in open_doc:

        # Step 1: Remove punctuations from words
        r_strip = review.translate(str.maketrans('','',string.punctuation))
        # Step 2: Convert words to lower case
        testset.append(r_strip.lower())
    open_doc.close()

    output_prob = []

    for sentence in testset:
        sentence = sentence.split(" ")
        class_prob = []
        for c in C:
            sum_c = logprior[c] 
            for i in range(len(sentence)):
                word = sentence[i]
                if word in V:
                    sum_c += loglikelihood[c][word]
            class_prob.append(sum_c)
        output_prob.append(np.argmax(class_prob))

    for i in range(len(output_prob)):
        output_prob[i] = str(output_prob[i])

    return output_prob

def main():
    nb_train = train_nb(doc,f_c)
    filename = input("Please enter the path to your file:\n")
    nb_test = test(filename, nb_train[0], nb_train[1], f_c, split_BoW)

    fw = open("results_file.txt", "w")
    fw.write("\n".join(nb_test))
    fw.close()

main()