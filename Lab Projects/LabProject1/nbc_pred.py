import numpy as np
import math
import string

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