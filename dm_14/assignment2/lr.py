#coding:utf-8

import numpy as np
import pickle
from collections import Counter

class SGDLR(object):
    def fit(self,X,y,iter,learning_rate):
        self.rate = learning_rate
        n_sample,n_feature = X.shape
        theta = np.ones(n_feature+1)
        X = np.column_stack((np.ones(n_sample),X))
        for i in range(iter):
            for i in range(n_sample):
                h = 1/(1.0+np.power(np.e,-np.dot(theta,X[i].T)))
                theta = theta - self.rate*(h-y[i])*X[i]
        self.theta = theta

    def predict(self,X):
        n_sample,n_feature = X.shape
        X = np.column_stack((np.ones(n_sample),X))
        h = 1/(1.0+np.power(np.e,-np.dot(X,self.theta.T)))
        return h
        

class MultiClassifier(object):
    def fit(self,X,y,iter,learning_rate):
        labels_counter = Counter(y)
        labels = labels_counter.keys()
        n_label = len(labels)
        self.sgdlr_classifiers = list()
        for i in range(n_label):
            sgdlr = SGDLR()
            label = labels[i]
            s_y = np.zeros(len(y))
            for i,value in enumerate(y):
                if value == label:
                    s_y[i] = 1.0
            sgdlr.fit(X,s_y,iter,learning_rate)
            self.sgdlr_classifiers.append(sgdlr)
        self.labels = labels 

    def predict(self,X):
        predict_labels = list()
        rs = list()
        for sgdlr in self.sgdlr_classifiers:
            r = sgdlr.predict(X)
            rs.append(r)
        rs = np.array(rs)
        rs = rs.T
        for values in rs:
            values = list(values)
            max_value = max(values)
            index = values.index(max_value)
            predict_labels.append(self.labels[index])
        return predict_labels


def kfoldcross(a,fold):
    n_sample = len(a)
    piece = n_sample/fold
    train_index = list()
    test_index = list()
    for i in range(fold):
        all = range(n_sample)
        if i < fold-1:
            test = range(i*piece,(i+1)*piece)
        else:
            test = range(i*piece,n_sample)
        train = list(set(all)-set(test))
        train_index.append(train)
        test_index.append(test)
    return zip(train_index,test_index)

def test_classifier(classifier,features,labels,iter=5,learning_rate=0.1):
    kfc = kfoldcross(labels,10)
    accuracies = list()
    for train_index,test_index in kfc:
        X = features[train_index]
        y = labels[train_index]
        classifier.fit(X,y,iter,learning_rate)
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

def regularization(X):
    max_values = np.max(X,axis=0)
    min_values = np.min(X,axis=0)
    X = (X-min_values) / (max_values-min_values)
    return X

def main():
    with open('lily.pickle') as f:
        lily = pickle.load(f)

    labels = lily['labels']
    tf = lily['tf']
    tf = tf.todense()
    tf = np.array(tf)
    tfidf = lily['tfidf']
    tfidf = tfidf.todense()
    tfidf = np.array(tfidf)
    #tfidf = regularization(tfidf)

    mulclassifier = MultiClassifier()
    test_classifier(mulclassifier,tf,labels,iter=5,learning_rate=0.1)
    test_classifier(mulclassifier,tfidf,labels,iter=5,learning_rate=5.0)

if __name__ == "__main__":
    main()
