from sklearn.linear_model import SGDClassifier

X = [[0,0],[1,1],[2,2]]
y = [0,1,2]   
clf = SGDClassifier(loss='log').fit(X,y)
result = clf.predict_proba([1,1])
print result

