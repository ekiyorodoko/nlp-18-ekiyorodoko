import sys
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model.logistic import LogisticRegression
import nltk


def main():
    # Read from standard input
    input_ = sys.argv
    classifier_type, version, input_ = input_[1], input_[2], input_[3]

    classifier = GaussianNB()
    
    #  Load data using pandas dataframe
    data = pd.read_table("dataset.txt", sep='\t', header=None, names=['text', 'label'])
    X_test = pd.read_table(input_, header=None, names=['input'])

    #   convert string labels to integers
    func = lambda x : [int(y) for y in x]
    func(data['label'])

    if classifier_type == "lr":
        classifier = LogisticRegression()
        
    if version == "n":    
        data['text'] = normalize(data['text'])
        X_test = normalize(X_test['input'])
    

    # tokenization: convert text document to matrix of tokens
    count_vect = CountVectorizer()  

    counts = count_vect.fit_transform(data['text'])
    X_test = count_vect.transform(X_test).toarray()
    

    # Split data into training and testing
    X_train = counts.toarray()
    y_train = data['label']
  

    # Predict labels for test documents
    train = classifier.fit(X_train, y_train)
    y_pred = train.predict(X_test)

    # write output to file
    fw = open("results"+classifier_type+version+".txt", "w")
    write = lambda x : [fw.write(str(y)+"\n") for y in x ]
    write(y_pred)
    fw.close()


# Normalize document text
def normalize(dataset):
    # casefolding: convert all characters in document to lower case
    dataset = dataset.map(lambda x: x.lower())

    # remove all punctuation
    rem_punc = lambda x : [y.translate(str.maketrans('','',string.punctuation)) for y in x]
    dataset = rem_punc(dataset)
    
    return dataset


main()

