from numpy import *
import math

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
	vocabSet = {}
	for document in dataSet:
		vocabSet = set(vocabSet) | set(document)
	return list(vocabSet)

# vocabList = createVocabList(loadDataSet()[0])
# inputSet = loadDataSet()[0][0]
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:	returnVec[vocabList.index(word)] = 1
		else:	print("Word '"+word+"' does not exist in vocabulary")
	return returnVec

def createTrainingMatrix(dataSet):
	trainMat = []
	for postinDoc in dataSet:
		trainMat.append(setOfWords2Vec(createVocabList(dataSet), postinDoc))
	return trainMat

# print(tra)
def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	p0Num = zeros(numWords);
	p1Num =zeros(numWords)
	p0Denom = 0.0
	p1Denom =0.0
	for i in range(numTrainDocs):
		if(trainCategory[i] == 1):
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = p1Num/p1Denom
	p0Vect = p0Num/p0Denom
	return p0Vect,p1Vect, pAbusive


def classify(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify*p1Vec) + log(pClass1)
	p0 = sum(vec2Classify*p0Vec) + log(1.0 - pClass1)
	if(p1 > p0):
		return 1
	else:
		return 0 

trainMatrix = createTrainingMatrix(loadDataSet()[0])
# print(trainMatrix)
trainCategory = loadDataSet()[1]
myVocabList = createVocabList(loadDataSet()[0])
print(myVocabList)
listClasses = loadDataSet()[1]

p0V, p1V, pAb = trainNB0(trainMatrix, trainCategory)

testEntry = ['love', 'my', 'dalmation']
thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
# print(classify(thisDoc, p0V, p1V, pAb))


testEntry = ['stupid', 'garbage', 'dalmation']
thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
# print(classify(thisDoc, p0V, p1V, pAb))