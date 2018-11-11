#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math
import numpy as np
import string
import nbc_pred as pred
import nbc_train as train
import data_wrangler as dw


# In[4]:


paths = ["dataset/amazon_cells_labelled.txt" ]
# ["dataset/imdb_labelled.txt","dataset/yelp_labelled.txt",

# In[5]:


dataset = dw.pre_process(paths)


# In[6]:


# Select documents of reviews from dataset
np_dataset = np.array(dataset).T
doc = np_dataset[0]


# In[7]:


# Seperate classes from the dataset
classes = np_dataset[1]
unique, counts = np.unique(classes, return_counts = True)


# In[8]:


# class categories
C = dict(zip(unique, counts))


# In[9]:


# convert corpus to a string of words and then a list of words
BoW = " ".join(doc).split(" ")


# In[10]:


# remove empty strings in bag of words
if '' in BoW:
    BoW.remove('')


# In[11]:


# Make list of words unique
u_words, count = np.unique(BoW, return_counts = True)
V = dict(zip(u_words, count))
print(len(u_words))
exit()


# In[12]:


def main():
    nb_train = train.train_nb(C,V, doc, dataset)
    print(nb_train[0])
    exit()
    filename = input("Please enter the path to your file:\n")
    try:
        nb_test = pred.test(filename, nb_train[0], nb_train[1], C, V)
    except FileNotFoundError:
        print("The file does not exist")
        exit()
    fw = open("results_file.txt", "w")
    fw.write("\n".join(nb_test))
    fw.close()


# In[ ]:


main()

