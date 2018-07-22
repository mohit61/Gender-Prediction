# Gender Prediction using 5 different classifiers

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
'female', 'male', 'male']


#1 Using Decision Tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
DecisionTree_prediction = clf.predict([[190, 70, 43]])
print("Decision Tree Prediction: ", DecisionTree_prediction)


#2 Using KNeighbors
clf = KNeighborsClassifier(3)
clf = clf.fit(X,Y)
KneighborsClassifier_prediction = clf.predict([[190, 70, 43]])
print("KneighborsClassifier_prediction: ", DecisionTree_prediction)


#3 Using GaussianNB
clf = GaussianNB()
clf = clf.fit(X,Y)
GaussianNB_prediction = clf.predict([[190, 70, 43]])
print("GaussianNB Prediction: ", DecisionTree_prediction)


#4 Using RandomForest
clf = RandomForestClassifier()
clf = clf.fit(X,Y)
RandomForestClassifier_prediction = clf.predict([[190, 70, 43]])
print("RandomForestClassifier Prediction: ", RandomForestClassifier_prediction)


#Using LDA
clf = LinearDiscriminantAnalysis()
clf = clf.fit(X,Y)
LDA_prediction = clf.predict([[190, 70, 43]])
print("LDA Prediction: ", LDA_prediction)
