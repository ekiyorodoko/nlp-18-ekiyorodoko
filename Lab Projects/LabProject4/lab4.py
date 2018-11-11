import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def main():
    # Read from standard input
    input_ = sys.argv
    classifier, version, output_file = input_[1], input_[2], input_[3]
    
    if classifier == "nb":
        #  Load data using pandas dataframe
        data = pd.read_table("dataset.txt", sep='\t', header=None, names=['text', 'label'])
        if version == "u":    
            X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=69)
        else:
            normalize()


# Normalize document text
def normalize(dataset):
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=69)

# count_vect = CountVectorizer()  
# counts = count_vect.fit_transform(data['text'])  

print(len(y_train))