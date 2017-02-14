from os import walk
from os import path
from random import shuffle
import math
import tensorflow as tf
from PIL import Image
from numpy import array
import numpy as np
import scipy.misc
import imageManipulationUtil
import dataManipulationUtil
import pickle

class genericDataSetLoader:

    basePath = "dataset"
    numClasses = 2
    numChannels = 3
    imageXSize = 224
    imageYSize = 224
    splitPercentage = 0.8
    filePaths = []
    totalDataY = []
    className2ClassIndxMap = {}
    trainingDataX = None
    testingDataX = None
    trainingDataY = None
    testingDataY = None
    trainingDataOffset = 0
    alreadySplitInTrainTest = False

    def __init__(self,alreadySplitInTrainTest,basePath,numClasses,splitPercentage,imageSizeX,imageSizeY):
        self.basePath = basePath
        self.numClasses = numClasses
        self.splitPercentage = splitPercentage
        self.imageXSize = imageSizeX
        self.imageYSize = imageSizeY
        self.alreadySplitInTrainTest = alreadySplitInTrainTest

    def __initializeClass2IndxMap(self,classList):
        idx=0
        for clazz in classList:
            self.className2ClassIndxMap[clazz]=idx
            idx=idx+1
        print self.className2ClassIndxMap

    def __getClassIndex(self,clazz):
        return self.className2ClassIndxMap[clazz]

    def __loadImageData(self,fileNames):
        imagesDataList = []
        cnt=0
        totalCnt = len(fileNames)
        for fName in fileNames:
            cnt = cnt+1 
            print "Loading Image:"+cnt+"/"+totalCnt
            imageDataAsArray = imageManipulationUtil.loadImageAsArray(fName)
            imageDataAsArray = imageManipulationUtil.squashImageArray(imageDataAsArray,self.imageXSize,self.imageYSize)
            imagesDataList.append(imageDataAsArray)
        img_np = np.array(imagesDataList)
        return img_np

    def __convertLabelsToOneHotVector(self,labelsList,numClasses):
        labelsArray = np.array(labelsList)
        oneHotVector = np.zeros((labelsArray.shape[0],numClasses))
        oneHotVector[np.arange(labelsArray.shape[0]), labelsArray] = 1
        return oneHotVector

    def __shuffle(self,list1,list2):
        list1_shuf = []
        list2_shuf = []
        index_shuf = range(len(list1))
        shuffle(index_shuf)
        for i in index_shuf:
            list1_shuf.append(list1[i])
            list2_shuf.append(list2[i])
        return list1_shuf,list2_shuf

    def __trainTestSplit(self,filePaths,labels,splitPercentage):
        splitIndex = int(math.ceil(splitPercentage*len(filePaths)))
        trainingDataX = filePaths[:splitIndex]
        trainingDataY = labels[:splitIndex]
        testingDataX = filePaths[splitIndex:]
        testingDataY = labels[splitIndex:]
        return trainingDataX,trainingDataY,testingDataX,testingDataY

    def prepareDataSetFromImages(self):
        if(self.alreadySplitInTrainTest):
            print "Data is already split into training,testing.Loading data..."
            self.__prepareDataSetFromAlreadySplitImages()
        else:
            print "Loading data after splitting and shuffling..."
            self.__prepareDataSetFromImagesSplitShuffle()

    def __prepareDataSetFromAlreadySplitImages(self):
        trainTestDirectory = next(walk(self.basePath))[1]
        if(len(trainTestDirectory)!=2):
            raise Exception("Number of split found more than 2. Expect only Train/Test")
        trainingDirs = next(walk(self.basePath+"/training"))[1]
        testingDirs = next(walk(self.basePath+"/testing"))[1]
        #print "Training directories:"+(str(trainingDirs))
        #print "Testing directories:"+(str(testingDirs))
        self.__initializeClass2IndxMap(trainingDirs)
        self.__initializeClass2IndxMap(testingDirs)
        self.trainingDataX = []
        self.trainingDataY = []
        self.testingDataX = []
        self.testingDataY = []
        for trainingDir in trainingDirs:
            trainingClassFiles = next(walk(path.join(self.basePath+"/training",trainingDir)))[2]
            for fName in trainingClassFiles:
                self.trainingDataX.append(self.basePath+"/training"+"/"+trainingDir+"/"+fName)
                self.trainingDataY.append(self.__getClassIndex(trainingDir))

        print "Shuffling the training dataset..."
        self.trainingDataX,self.trainingDataY = self.__shuffle(self.trainingDataX,self.trainingDataY)

        for testingDir in testingDirs:
            testingClassFiles = next(walk(path.join(self.basePath+"/testing",testingDir)))[2]
            for fName in testingClassFiles:
                self.testingDataX.append(self.basePath+"/testing"+"/"+testingDir+"/"+fName)
                self.testingDataY.append(self.__getClassIndex(testingDir))

        print "Shuffling the testing dataset..."
        self.testingDataX,self.testingDataY = self.__shuffle(self.testingDataX,self.testingDataY)

        self.__postProcessData()
        print self.testingDataX.shape
        print self.testingDataY
        print self.trainingDataX.shape
        print self.trainingDataY
        self.__save()

    def __prepareDataSetFromImagesSplitShuffle(self):
        self.trainingDataOffset = 0
        classDirectories = next(walk(self.basePath))[1]
        if(len(classDirectories)!=self.numClasses):
            raise Exception("Number of classes found in dataset is not equal to the specified numClasses")
        self.__initializeClass2IndxMap(classDirectories)
        for classDirectory in classDirectories:
            classFiles = next(walk(path.join(self.basePath,classDirectory)))[2]
            for fname in classFiles:
                self.filePaths.append(self.basePath+"/"+classDirectory+"/"+fname)
                self.totalDataY.append(self.__getClassIndex(classDirectory))

        print "Shuffling the dataset..."
        #shuffle the dataset for randomization
        shuffledFilePaths,shuffledLabels = self.__shuffle(self.filePaths,self.totalDataY)

        #split into train-test
        print "Splitting into train and test..."
        self.trainingDataX,self.trainingDataY,self.testingDataX,self.testingDataY = self.__trainTestSplit(shuffledFilePaths,shuffledLabels,self.splitPercentage)

        self.__postProcessData()
        self.__save()

    def __postProcessData(self):
        #convert file paths into numpy array by reading the files
        print "Reading the training image files..."
        self.trainingDataX = self.__loadImageData(self.trainingDataX)
        print "Reading the training image files..."
        self.testingDataX = self.__loadImageData(self.testingDataX)

        #convert class lables into one hot encoded
        print "Creating one hot encoded vectors for training labels..."
        self.trainingDataY = self.__convertLabelsToOneHotVector(self.trainingDataY,self.numClasses)
        print "Creating one hot encoded vectors for testing labels..."
        self.testingDataY = self.__convertLabelsToOneHotVector(self.testingDataY,self.numClasses)


    def loadData(self):
        pklFile = open("preparedData.pkl", 'rb')
        preparedData=pickle.load(pklFile)
        self.trainingDataX = preparedData["trainingX"]
        self.trainingDataY = preparedData["trainingY"]
        self.testingDataX = preparedData["testingX"]
        self.testingDataY = preparedData["testingY"]
        print "Data loaded..."
        print self.trainingDataX.shape
        print self.trainingDataY.shape
        print self.testingDataX.shape
        print self.testingDataY.shape

    def __save(self):
        preparedData={}
        preparedData["trainingX"] = self.trainingDataX
        preparedData["trainingY"] = self.trainingDataY
        preparedData["testingX"] = self.testingDataX
        preparedData["testingY"] = self.testingDataY
        pklFile = open("preparedData.pkl", 'wb')
        pickle.dump(preparedData, pklFile)
        pklFile.close()

    def getNextTrainBatch(self,batchSize):
        trainDataX = dataManipulationUtil.selectRows(self.trainingDataX,self.trainingDataOffset,batchSize)
        trainDataY = dataManipulationUtil.selectRows(self.trainingDataY,self.trainingDataOffset,batchSize)
        self.trainingDataOffset = self.trainingDataOffset+batchSize
        return trainDataX,trainDataY

    def resetBatch(self):
        self.trainingDataOffset=0

    def getNextTestBatch(self):
        return self.testingDataX,self.testingDataY
