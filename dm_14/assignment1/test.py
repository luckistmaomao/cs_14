#coding:utf-8

"""
anthor:yuzt <zhenting.yu@gmail.com>
operating system:ubuntu 12.04
python version:2.7.3
"""

import random
import os
import numpy as np
from collections import Counter
from copy import deepcopy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import StratifiedKFold
import pickle
from scipy.sparse import coo_matrix

def cal_tf_and_tfidf(corpus):
    vectorizer = CountVectorizer(min_df=1)
    M = vectorizer.fit_transform(corpus)
    tf = M.toarray()
    transformer = TfidfTransformer()
    N = transformer.fit_transform(tf)
    tfidf = N.toarray()
    return tf,tfidf

def main():
    lily_objs = list()
    with open('lily.txt') as f:
        for line in f:
            line = line.strip() 
            lily_obj = dict()
            label,content = line.split('\t')
            lily_obj['label'] = label
            lily_obj['words'] = content.split(' ')
            lily_objs.append(lily_obj)

    random.seed(0)
    random.shuffle(lily_objs)    #随机打乱

    corpus = [' '.join(lily_obj['words']) for lily_obj in lily_objs]
    tf,tfidf = cal_tf_and_tfidf(corpus)

    labels = [lily_obj['label'] for lily_obj in lily_objs]
    labels = np.array(labels)

    lily = dict()

    tf = coo_matrix(tf)
    tfidf = coo_matrix(tfidf)
    lily['tf'] = tf
    lily['tfidf'] = tfidf
    lily['labels'] = labels

    with open('lily.pickle','wb') as f:
        pickle.dump(lily,f)

    tf = np.array(tf)
    print type(tf)
    

if __name__ == "__main__":
    main()
