from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # print(curLine)
        fltLine0, fltLine1 = float(curLine[0]), float(curLine[1])
        dataMat.append([fltLine0, fltLine1])
        # print(fltLine0)
    return dataMat

def plotTestSet():
    dataSet = loadDataSet('testSet.txt')
    x, y = [], []
    for i in range(0, len(dataSet)):
        x.append(dataSet[i][0])
        y.append(dataSet[i][1])
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax =fig.add_subplot()
    ax.scatter(x,y)
    ax.plot()
    plt.show()

# plotTestSet()


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

# dataSet = matrix(loadDataSet('testSet.txt'))
# print(dataSet)
# print(distEclud(dataSet[0], dataSet[1]))
# print(randCent(dataSet, 2))

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        # print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

# possible markers + v s D

def plotDifferentClusters():
    dataSet = matrix(loadDataSet('testSet.txt'))
    centroids, clusters = kMeans(dataSet, 4)
    # print(clusters)
    # print(clusters[0][0,0])
    x0, x1, x2, x3, y0, y1, y2, y3= [], [], [], [], [], [], [], []
    for i in range(len(clusters)):
        if clusters[i][0,0] == 0:
            x0.append(dataSet[i,0]); y0.append(dataSet[i,1])
        if clusters[i][0,0] == 1:
            x1.append(dataSet[i,0]); y1.append(dataSet[i,1])
        if clusters[i][0,0] == 2:
            x2.append(dataSet[i,0]); y2.append(dataSet[i,1])
        if clusters[i][0,0] == 3:
            x3.append(dataSet[i,0]); y3.append(dataSet[i,1])
    cenX, cenY = [], []
    for i in range(len(centroids)):
        cenX.append(centroids[i,0]); cenY.append(centroids[i,1])
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax =fig.add_subplot()
    ax.scatter(x0,y0)
    ax.scatter(x1,y1)
    ax.scatter(x2,y2)
    ax.scatter(x3,y3)
    ax.scatter(cenX, cenY, marker = "+", color = 'black')
    ax.plot()
    plt.show()

# plotDifferentClusters()



def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0)
    print(centroid0)
    centList =[centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print ("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print ('the bestCentToSplit is: ',bestCentToSplit)
        print ('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
    return mat(centList), clusterAssment

# dataSet3 = mat(loadDataSet("testSet2.txt"))
# biKmeans(dataSet3, 3)