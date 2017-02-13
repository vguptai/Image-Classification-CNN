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

    def __init__(self,basePath,numClasses,splitPercentage,imageSizeX,imageSizeY):
        self.basePath = basePath
        self.numClasses = numClasses
        self.splitPercentage = splitPercentage
        self.imageXSize = imageSizeX
        self.imageYSize = imageSizeY

    def __initializeClass2IndxMap(self,classList):
        idx=0
        for clazz in classList:
            self.className2ClassIndxMap[clazz]=idx
            idx=idx+1

    def __getClassIndex(self,clazz):
        return self.className2ClassIndxMap[clazz]

    def __loadImageData(self,fileNames):
        imagesDataList = []
        for fName in fileNames:
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

    def loadData(self):
        classDirectories = next(walk(self.basePath))[1]
        if(len(classDirectories)!=self.numClasses):
            raise Exception("Number of classes found in dataset is not equal to the specified numClasses")
        self.__initializeClass2IndxMap(classDirectories)
        for classDirectory in classDirectories:
            classFiles = next(walk(path.join(self.basePath,classDirectory)))[2]
            for fname in classFiles:
                self.filePaths.append(self.basePath+"/"+classDirectory+"/"+fname)
                self.totalDataY.append(self.__getClassIndex(classDirectory))

        #shuffle the dataset for randomization
        shuffledFilePaths,shuffledLabels = self.__shuffle(self.filePaths,self.totalDataY)
        #split into train-test
        trainingDataX,trainingDataY,testingDataX,testingDataY = self.__trainTestSplit(shuffledFilePaths,shuffledLabels,self.splitPercentage)
        #convert file paths into numpy array by reading the files
        trainingDataX = self.__loadImageData(trainingDataX)
        testingDataX = self.__loadImageData(testingDataX)
        trainingDataY = self.__convertLabelsToOneHotVector(trainingDataY,self.numClasses)
        testingDataY = self.__convertLabelsToOneHotVector(testingDataY,self.numClasses)
        return trainingDataX,trainingDataY,testingDataX,testingDataY
