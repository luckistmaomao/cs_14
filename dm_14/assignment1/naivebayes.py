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


class NBD(object):
    """
    fit(X,y) 用来训练，X是特征矩阵，y是与X对应的类标签
    predit(X) 用来预测，X是特征矩阵，返回预测的类标签
    """
    def fit(self,X,y):
        labels_counter = Counter(y)
        labels = labels_counter.keys()
        n_feature = len(X[0])
        n_label = len(labels)
        feature_count = np.ones((n_label,n_feature))

        for i,label in enumerate(labels):
            tmp = X[y == label,:]
            feature_count[i,:] += np.sum(tmp,axis=0)
        label_count = feature_count.sum(axis=1)
        self.labels = labels
        self.feature_log_prob = np.log(feature_count)-np.log(label_count.reshape(-1,1))
        self.label_log_prob = np.log(label_count/sum(label_count*1.0))

    def predict(self,X):
        predict_labels = list()
        X = np.array(X)
        for x in X:
            probs =  np.dot(self.feature_log_prob,x) + self.label_log_prob
            probs = list(probs)
            max_prob = max(probs)
            predict_label = self.labels[probs.index(max_prob)]
            predict_labels.append(predict_label)
        return predict_labels


class NBCD(object):
    """
    fit(X,y) 用来训练，X是特征矩阵，y是与X对应的类标签
    predit(X) 用来预测，X是特征矩阵，返回预测的类标签
    discretize(X) 用来对连续特征离散化，X是特征举证，返回离散化后的特征矩阵
    """
    def fit(self,X,y):
        X = np.array(X)
        labels_counter = Counter(y)
        labels = labels_counter.keys()
        X = self.discretize(X)
        n_label = len(labels)
        n_feature = X.shape[1]

        feature_count = np.ones((n_label,n_feature))
        for i,label in enumerate(labels):
            tmp = X[y == label,:]
            feature_count[i,:] += np.sum(tmp,axis=0)

        label_count = feature_count.sum(axis=1)
        self.labels = labels
        self.feature_log_prob = np.log(feature_count)-np.log(label_count.reshape(-1,1))
        self.label_log_prob = np.log(label_count/sum(label_count*1.0))
        
    def predict(self,X):
        predict_labels = list()
        X = np.array(X)
        X = self.discretize(X)
        for x in X:
            probs =  np.dot(self.feature_log_prob,x) + self.label_log_prob
            probs = list(probs)
            max_prob = max(probs)
            predict_label = self.labels[probs.index(max_prob)]
            predict_labels.append(predict_label)
        return predict_labels
        
    def discretize(self,X):
        X = X*100
        X = np.floor(X)
        return X
        

class NBCG(object):
    """
    fit(X,y) 用来训练，X是特征矩阵，y是与X对应的类标签
    predit(X) 用来预测，X是特征矩阵，返回预测的类标签
    """
    def fit(self,X,y):
        X = np.array(X)
        labels_counter = Counter(y)
        labels = labels_counter.keys()
        labels = np.array(labels)
        label_count = labels_counter.values()
        label_count = np.array(label_count)
        
        n_label = len(labels)
        n_feature = X.shape[1]
        feature_mean = np.zeros((n_label,n_feature))
        feature_var = np.zeros((n_label,n_feature))
        smooth_delta = 1e-9
        for i,label in enumerate(labels):
            tmp = X[y == label,:]
            feature_mean[i,:] = np.mean(tmp,axis=0)
            feature_var[i,:] = np.var(tmp,axis=0) + smooth_delta
            
        self.n_label = n_label
        self.n_feature = n_feature
        self.labels = list(labels)
        self.feature_mean = feature_mean
        self.feature_var = feature_var 
        self.label_log_prob = np.log(label_count/sum(label_count*1.0))
    
    def predict(self,X):
        predict_labels = list()
        X = np.array(X)
        n_label = self.n_label
        n_feature = self.n_feature
        for x in X:
            mat = [deepcopy(x) for i in range(n_label)]
            mat = np.array(mat)
            feature_log_prob = -0.5*np.log(2*np.pi*self.feature_var) 
            feature_log_prob -= 0.5*((mat-self.feature_mean)**2)/self.feature_var 
            probs = np.sum(feature_log_prob,axis=1) + self.label_log_prob
            probs = list(probs)
            max_prob = max(probs)
            predict_label = self.labels[probs.index(max_prob)]
            predict_labels.append(predict_label)
        return predict_labels
    
def test_classifier(classifier,features,labels):
    skf = StratifiedKFold(labels,10)
    accuracies = list()
    for train_index,test_index in skf:
        X = features[train_index]
        y = labels[train_index]
        classifier.fit(X,y)
        predict_labels = classifier.predict(features[test_index])

        reuslt_labels = labels[test_index]
        count = 0
        for i in range(len(predict_labels)):
            if reuslt_labels[i] == predict_labels[i]:
                count += 1
        accuracy = count*1.0/len(predict_labels)
        accuracies.append(accuracy)
        print "Test case %s accuracy is %s" % (len(accuracies),accuracy)
    print 'mean:',np.mean(accuracies)
    print 'var:',np.var(accuracies)
    print 'std',np.std(accuracies)

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

    print "-----------------------------------------"
    print "Test NBD"
    nbd = NBD()
    test_classifier(nbd,tf,labels)
    print "-----------------------------------------"
    print "Test NBCD"
    nbcd = NBCD()
    test_classifier(nbcd,tfidf,labels)
    print "-----------------------------------------"
    print "Test NBCG"
    nbcg = NBCG()
    test_classifier(nbcg,tfidf,labels)
    print "-----------------------------------------"
    print "END"

if __name__ == "__main__":
    main()
