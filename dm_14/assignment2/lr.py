#coding:utf-8

import numpy as np
import pickle

class SGDLR(object):
    def fit(self,X,y):
        self.rate = 0.1
        n_sample,n_feature = X.shaple
        theta = np.zeros(n_feature+1)
        X = np.column_stack((np.ones(n_sample),X))
        for i in range(n_sample):
            theta = theta - self.rate*(1.0/(1+np.power(np.e,-theta.T*X[i])-y[i]))*X[i]
        self.theta = theta

    def predict(self,X):
        return 1.0/(1+np.power(np.e,-X*theta.T))
        
def one_vs_all():
    pass

def main():
    with open('lily.pickle') as f:
        lily = pickle.load(f)

    labels = lily['labels']
    tf = lily['tf']

    

if __name__ == "__main__":
    main()
