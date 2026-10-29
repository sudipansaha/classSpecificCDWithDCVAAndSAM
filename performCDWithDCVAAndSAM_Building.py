import torch
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pathlib import Path

from skimage.measure import label, regionprops
from skimage import morphology
from skimage.transform import resize
from skimage import filters

from sklearn.metrics import confusion_matrix, f1_score


from dcvaFunctionResnet18 import dcva

from segment_anything import sam_model_registry, SamPredictor

from samAndOneExampleSegmentation import segmentWithSamAndOneExampleImage

from preOrPostprocessing import saturateImage


### Set manual seeds
manualSeed = 62
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)


### DCVA related options

class DCVAOptionValues:
    def __init__(self):
        self.layersToProcess = '2,5,8'
        self.otsuScalingFactor = 1.25
        self.topPercentSaturationOfImageOk = False
        self.objectMinSize = 11
optDCVA = DCVAOptionValues()
optDCVA.layersToProcess = '2'
optDCVA.otsuScalingFactor = 1
optDCVA.topPercentSaturationOfImageOk = False
optDCVA.objectMinSize = 64

## number of example images to use
numOfExampleImagesToUse = 8


##Pixel values of interest
buildingPixelValue=[128,0,0]
treePixelValue = [0,255,0]
lowVegetationPixelValue = [0,128,0]
nonVegetationSurface = [128,128,128]


## Considering building to other changes
currentlyConsideredTypePixelValue = buildingPixelValue.copy()


## Path to the .npy file containing test images path
testPathFilePath = '../datasets/secondTestBuildingToOtherPathsSize50.npy'




## Reading the above file containing path to the .npy file containing test images path
testPathsList = np.load(testPathFilePath) 




### Example image (for SAM) path
exampleImagePathList = ['../datasets/buildingExampleImages/images/00042.png',
                         '../datasets/buildingExampleImages/images/00133.png',
                         '../datasets/buildingExampleImages/images/00210.png',
                         '../datasets/buildingExampleImages/images/00360.png',
                         '../datasets/buildingExampleImages/images/00415.png',
                         '../datasets/buildingExampleImages/images/00434.png',
                         '../datasets/buildingExampleImages/images/00529.png',
                         '../datasets/buildingExampleImages/images/00561.png',
                         '../datasets/buildingExampleImages/images/00687.png',
                         '../datasets/buildingExampleImages/images/00824.png']
exampleLabelPathList = []
for exampleImagePath in exampleImagePathList:
    exampleLabelPath = exampleImagePath.replace('images','labels')
    exampleLabelPathList.append(exampleLabelPath)
    
### Modify according to numOfExampleImagesToUse
exampleImagePathList = exampleImagePathList[0:numOfExampleImagesToUse]
exampleLabelPathList = exampleLabelPathList[0:numOfExampleImagesToUse]



### Reading the example images
exampleImageList =[]
exampleBinaryGtList =[]

for exampleImagePathIterator in exampleImagePathList:
    exampleImageTemp = skimage.io.imread(exampleImagePathIterator)
    exampleImageList.append(exampleImageTemp)
    
    
for exampleLabelPathIterator in exampleLabelPathList:
    exampleLabelImageTemp = plt.imread(exampleLabelPathIterator)
    
    ## This is in black object - white background format, we need to reverse it
    exampleLabelImageTemp = 1-exampleLabelImageTemp  
    

    exampleBinaryGtList.append(exampleLabelImageTemp)
                            
               


precisionArray = []
sensitivityArray = []
specificityArray= []
f1ScoreArray = []
iouChangedClassArray = []
iouValueMeanArray = []


for index,preChangeImagePath in enumerate(testPathsList):



    print('Processing image: '+str(index+1))

      
   
    postChangeImagePath = preChangeImagePath.replace('im1','im2')
    

    preChangeLabelPath = preChangeImagePath.replace('im1','label1')
    postChangeLabelPath = postChangeImagePath.replace('label1','label2')
    
    ### Load images 
    preChangeImage = skimage.io.imread(preChangeImagePath)
    postChangeImage = skimage.io.imread(postChangeImagePath)
    
    ### Normalize input images
    ### Images are initially in range 0-255
    preChangeImageProcessedForDCVA = saturateImage().standardizeRGBImagesWithImagenetValues(preChangeImage/255.)
    postChangeImageProcessedForDCVA = saturateImage().standardizeRGBImagesWithImagenetValues(postChangeImage/255.) 
    
        
    cdMagnitudeOutput = dcva(preChangeImageProcessedForDCVA,postChangeImageProcessedForDCVA,optDCVA)
    
    detectedChangeMapNormalized=(cdMagnitudeOutput-np.amin(cdMagnitudeOutput))/(np.amax(cdMagnitudeOutput)-np.amin(cdMagnitudeOutput))
    
    cdMap=np.zeros(detectedChangeMapNormalized.shape, dtype=bool)

    otsuThreshold=filters.threshold_otsu(detectedChangeMapNormalized)
    cdMap = (detectedChangeMapNormalized>optDCVA.otsuScalingFactor*otsuThreshold)
    cdMap=morphology.remove_small_objects(cdMap,min_size=optDCVA.objectMinSize)  
 
    
    
    
    ## Performing segmentation of prechange image
    preChangeSegmentationOutputList = []
    for segmentationIter in range(len(exampleImageList)):
        exampleImage = exampleImageList[segmentationIter]
        exampleBinaryGt = exampleBinaryGtList[segmentationIter]
        preChangeSegmentationOutputList.append(segmentWithSamAndOneExampleImage(exampleImage, exampleBinaryGt, preChangeImage,20))    
    
    preChangeSegmentationOutputComposition = np.zeros((cdMap.shape[0],cdMap.shape[1]),dtype='bool')
    for preChangeSegmentation in preChangeSegmentationOutputList:

        preChangeSegmentationOutputComposition = np.logical_or(preChangeSegmentationOutputComposition,preChangeSegmentation)
     
    
    
    
    
    ## Combining SAM based CD output with DCVA output
    classSpecificCDOutput = np.logical_and(cdMap,preChangeSegmentationOutputComposition)
    
 
    
    
    imageID = preChangeImagePath.split('/')[-1].split('.')[0]
    plt.imsave('./tempResultDirectory/result'+imageID+'.png',np.repeat(np.expand_dims((1-classSpecificCDOutput),2),3,2).astype(float))

    

    



