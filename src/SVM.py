import pandas as pd
from sklearn.svm import SVC
from random import sample
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np

def to_int_map(c):
    if c == 'A':
        return 0
    elif c == 'C':
        return 1
    elif c == 'D':
        return 2
    elif c == 'G':
        return 3
    elif c == 'N':
        return 4
    elif c == 'R':
        return 5
    elif c == 'S':
        return 6
    elif c == 'T':
        return 7
    else:
        print('ERROR!')
    return c

def solutions_map(c):
    #EI, IE, N
    if c == 'EI':
        return 0
    elif c == 'IE':
        return 1
    elif c == "N":
        return 2
    else:
        print("ERROR")
    return c

def accuracy(x,y):
    hits = 0
    size = len(x)
    if(size != len(y)):
        print("Error: size not equal")
        return -1
    for i in range(len(x)):
        if(x[i] == y[i]):
            hits += 1
    return hits/len(x)
    
def getRandSample(size, ratio):
    full = range(size)
    ratioSample = sample(full, int(size*ratio))
    outSample = []
    for item in full:
        if(item not in ratioSample):
            outSample.append(item)
    return [ratioSample, outSample]

"""
dataset = pd.read_csv('..\\datasetSPLICEDNA\\splice.data.csv', header=None, skiprows=1)
solutions = [solutions_map(tempo) for tempo in dataset[0]]
rawDNA = dataset[2]
DNA = []
for item in rawDNA:
    DNA += [[to_int_map(l) for l in ''.join(item.split())]]

# create Training and Testing sets
splitData = getRandSample(len(DNA), 0.8)
trainDNA = []
trainSol = []
testDNA = []
testSol = []
for index in splitData[0]:
    trainDNA.append(DNA[index])
    trainSol.append(solutions[index])
for index in splitData[1]:
    testDNA.append(DNA[index])
    testSol.append(solutions[index])

# SVM method
clf = SVC(kernel='poly', degree=8, gamma='auto')
# Use training set to fit model
clf.fit(trainDNA, trainSol)
# create prediction with testing set
pred = clf.predict(testDNA)
# compare testing predictions with solutions
print(accuracy(pred, testSol))
# Train and test on the same set
clf2 = SVC(kernel='poly', degree=8, gamma='auto')
clf2.fit(DNA, solutions)
pred2 = clf2.predict(DNA)
print(accuracy(pred2, solutions))

#
#
#
#
#
"""
from data_prep import (
  read_dataset_from_csv,
  preprocess_data_and_get_X_and_y,
  partition_into_training_and_testing)


newDataset = read_dataset_from_csv('..\\datasetSPLICEDNA\\splice.data.csv')
X, y = preprocess_data_and_get_X_and_y(newDataset)
X_train, X_test, y_train, y_test = partition_into_training_and_testing(X, y)

y_train_format = []
for item in y_train:
    if(item[0] == 1):
        y_train_format.append(0)
    elif(item[1] == 1):
        y_train_format.append(1)
    elif(item[2] == 1):
        y_train_format.append(2)
    else:
        print("FORMAT ERROR")

y_test_format = []
for item in y_test:
    if(item[0] == 1):
        y_test_format.append(0)
    elif(item[1] == 1):
        y_test_format.append(1)
    elif(item[2] == 1):
        y_test_format.append(2)
    else:
        print("FORMAT ERROR")    

#print(X_train[:5])
#print(y_train[0][0])
#print(y_train[0])

print(y_train_format[:10])

clf3 = SVC(random_state=0)#.fit(X_train, y_train_format)
#score = cross_val_score(clf3, X_train, y_train_format, cv=10, n_jobs=4)
#print(score)
#print(clf3.score(X_test, y_test_format))

param_grid = [
  {'C': [1, 10, 100, 1000], 
   'gamma': [0.01, 0.001, 0.0001, 'auto'], 
   'kernel': ['linear','poly','rbf'], 
   'degree':[2,3,4,5,6,7,8]},
 ]

grid = GridSearchCV(estimator=clf3, 
                    param_grid=param_grid,
                    return_train_score=True,
                    error_score='raise',
                    n_jobs=4,
                    cv=10
                    ).fit(X_train, y_train_format)

print(grid)
print(grid.score(X_test, y_test_format))

best_est = grid.best_estimator_
test_score = best_est.score(X_test, y_test_format)
print(test_score)
#clf3.fit(X_train, y_train_format)
  
print("done")

