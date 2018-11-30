#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
import re
from sklearn.feature_extraction.text import  TfidfVectorizer, CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from sklearn.linear_model import Perceptron



#Preprocesses the tweets
def prep(list_doc):
    tweets = []
    # TODO: load training data
    with open(list_doc,'r') as f:
        for line in f:
            tweet = line.lower().strip().split('\t')
            regex_url = r"(http|https|bit.ly|reut)\W\S+|(www. )\S+\b|\w+\S+(?=.com?|uk|org|info)\S+\b"
            regex_happy = r"[:=]+[D\)\]\}]+|(;)+[D\)\]\}]+|([:=]+[pPd3]+)|([;]+[pPd3]+)"    #Converts happy emoticons to the word Happy
            regex_sad = r"[:=]+[\(\[\{]+"               #Converts sad emoticons to the word Sad
            regex_long = r"(.)\1+"                      #Shortens elongated words
            subst = ""
            tweet[2] = re.sub(regex_url, subst, tweet[2])
            tweet[2] = re.sub(regex_long, "\\1\\1", tweet[2])
            tweet[2] = re.sub(regex_happy,"happy",tweet[2])
            tweet[2] = re.sub(regex_sad,"sad",tweet[2])

            tweets.append([tweet[1],tweet[2]])

    return(tweets)

#Creates list of Tweet IDs
def ids(data):
    id_list = []
    with open(data,'r') as f:
        for line in f:
            tweet = line.strip().split('\t')
            id_list.append(tweet[0])

    return(id_list)

train_set = prep("twitter-training-data.txt")

for classifier in ['CountVectorizer', 'TfidfVectorizer', 'HashingVectorizer']: # You may rename the names of the classifiers to something more descriptive
    if classifier == 'CountVectorizer':
        print('Training ' + classifier)
        #extract features for training CountVectorizer using the below parameters
        vect = CountVectorizer(ngram_range = (1,3),token_pattern = r"\b\w+\b", min_df = 8, stop_words='english')
        #trains sentiment classifier
        train_vect = vect.fit_transform([t[1] for t in train_set])

    elif classifier == 'TfidfVectorizer':
        print('Training ' + classifier)
        #extract features for training TfidfVectorizer using the below parameters
        vect = TfidfVectorizer(ngram_range = (1,3),token_pattern = r"\b\w+\b",stop_words='english',min_df = 8, sublinear_tf=True, use_idf=True)

        #trains sentiment classifier
        train_vect = vect.fit_transform([t[1] for t in train_set])

    elif classifier == 'HashingVectorizer':
        print('Training ' + classifier)
        #extract features for training HashingVectorizer using the below parameters
        vect = HashingVectorizer(ngram_range = (1,3),token_pattern = r"\b\w+\b",stop_words='english')

        #trains sentiment classifier
        train_vect = vect.fit_transform([t[1] for t in train_set])


    for testset in testsets.testsets:
        #classifies tweets in test set
        test = prep(testset)
        test_features = vect.transform([t[1] for t in test])

        #Depending on the sentiment classifier, a correpsonding tweet classifier is used to obtained the best results.
        if classifier == "HashingVectorizer":
            classif = Perceptron(max_iter=300)
            classif.fit(train_vect,[(t[0]) for t in train_set])
        elif classifier == "TfidfVectorizer":
            classif = BernoulliNB()
            classif.fit(train_vect,[(t[0]) for t in train_set])
        else:
            classif = MultinomialNB()
            classif.fit(train_vect,[(t[0]) for t in train_set])

        #Predicts the sentiment type of the test features based on the training classifiers
        predictions = classif.predict(test_features)
        id_list = ids(testset)                  #calls the ids function for list of Tweet IDs
        id_list_predict = list(zip(id_list,list(predictions)))
        diction = dict(id_list_predict)         #Creates a dictionary of Tweet IDs and corresponding sentiment


        evaluation.evaluate(diction, testset, classifier)

        evaluation.confusion(diction, testset, classifier)
