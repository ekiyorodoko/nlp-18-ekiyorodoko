import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main():
    # Read from standard input
    input_ = sys.argv
    classifier_type, version, output_file = input_[1], input_[2], input_[3]
    
    if classifier_type == "nb":
        classifier = GaussianNB()
        #  Load data using pandas dataframe
        data = pd.read_table("dataset.txt", sep='\t', header=None, names=['text', 'label'])
        if version == "n":    
            normalize(data)
        
        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=69)

        # Predict labels for test documents
        y_pred = classifier.fit(X_train, y_train).predict(X_test)

    else: 
        if version = "n":
            #
        else:
            #

# Normalize document text
def normalize(dataset):
    # casefolding: convert all characters in document to lower case
    dataset['text'] = dataset.text.map(lambda x: x.lower())

    # remove all punctuation
    dataset['text'] = dataset.text.replace('[^\w\s]', '')

    # tokenization: convert text into single words


    X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['label'], test_size=0.2, random_state=19)

# count_vect = CountVectorizer()  
# counts = count_vect.fit_transform(data['text']) 


