from genericDataSetLoader import *
from config import *

genericDataSetLoader = genericDataSetLoader(True,"dataset",n_classes,testTrainSplit,imageSizeX,imageSizeY)
genericDataSetLoader.prepareDataSetFromImages()
