from genericDataSetLoader import *
from config import *

genericDataSetLoader = genericDataSetLoader(False,"dataset",n_classes,testTrainSplit,imageSizeX,imageSizeY)
genericDataSetLoader.prepareDataSetFromImages()
