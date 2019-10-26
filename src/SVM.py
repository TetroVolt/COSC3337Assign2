import pandas as pd
from sklearn.svm import SVC
from random import sample

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
clf = SVC(gamma='auto')
# Use training set to fit model
clf.fit(trainDNA, trainSol)
# create prediction with testing set
pred = clf.predict(testDNA)
# compare testing predictions with solutions
print(accuracy(pred, testSol))

# Train and test on the same set
clf2 = SVC(gamma='auto')
clf2.fit(DNA, solutions)
pred2 = clf2.predict(DNA)
print(accuracy(pred2, solutions))
