#!/usr/bin/env python

from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1], [1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#use KNN to classify
def classify(input, dataSet, label, k):
    dataSize = dataSet.shape[0]
    #calculate the distance
    diff = tile(input, (dataSize,1)) - dataSet
    sqdiff = diff ** 2
    squareDist = sum(sqdiff, axis=1) 
    dist = squareDist ** 0.5

    #sort the distance
    sortedDistIndex = argsort(dist)

    classCount = {}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1

    maxCount = 0
    for key,value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key
    return classes

group, labels = createDataSet()
input = array([0.2,0.2])
output = classify(input, group, labels, 3)
print("input is:", input, "result is:", output)
