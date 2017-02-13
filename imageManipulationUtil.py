from PIL import Image
from numpy import array
import numpy as np
import scipy.misc

def squashImageArray(imageDataArray,sizeX,sizeY):
    return scipy.misc.imresize(imageDataArray,(sizeX,sizeY))

def loadImageAsArray(imagePath):
    imageData = Image.open(imagePath)
    imageData.load()
    return np.asarray(imageData)

def testSquashing():
    imageData = loadImageAsArray("test.jpg")
    imageDataScaled = squashImageArray(imageData,224,224)
    img1 = Image.fromarray(imageDataScaled, 'RGB')
    img1.show()
    img2 = Image.fromarray(imageData, 'RGB')
    img2.show()
