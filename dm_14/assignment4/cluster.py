#coding:utf-8

from sklearn.cluster import KMeans
import pickle
import numpy as np

#class KMeans(object):
#    pass

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

    kmeans = KMeans(n_clusters=10)
    a = kmeans.fit_predict(tf)
    print len(a)


if __name__ == "__main__":
    main()
