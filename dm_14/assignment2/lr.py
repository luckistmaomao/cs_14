#coding:utf-8

import numpy as np
import pickle
from collections import Counter
from sklearn.cross_validation import StratifiedKFold

class SGDLR(object):
    def fit(self,X,y):
        self.rate = 10
        n_sample,n_feature = X.shape
        theta = np.ones(n_feature+1)
        X = np.column_stack((np.ones(n_sample),X))
        for i in range(n_sample):
            theta = theta - self.rate*(1.0/(1.0+np.power(np.e,-np.dot(theta.T,X[i])))-y[i])*X[i]
        self.theta = theta

    def predict(self,X):
        n_sample,n_feature = X.shape
        X = np.column_stack((np.ones(n_sample),X))
        return 1.0/(1.0+np.power(np.e,-np.dot(X,self.theta.T)))
        
class MultiClassifier(object):
    def fit(self,X,y):
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
            sgdlr.fit(X,s_y)
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

    skf = StratifiedKFold(labels,10,shuffle=True)

    classifier = MultiClassifier()
    features = tf
    accuracies = list()
    for train_index,test_index in skf:
        X = tf[train_index]
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

if __name__ == "__main__":
    main()
