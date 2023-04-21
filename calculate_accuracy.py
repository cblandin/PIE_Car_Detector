import numpy as np
import os
from project_common import getOverlap

class outputData :

    def __init__(self, outputFileName):

        outputFile = open(outputFileName, 'r')
        outputLines = outputFile.read().splitlines()
        outputFile.close()
        self.data = [line.split(", ") for line in outputLines]

    def getFrameData(self, i):
        splitFrameData = self.data[i]
        numDetections = int(len(splitFrameData) / 5)
        frameDetectionsArray = -np.ones((numDetections, 5))
        for i in np.arange(numDetections):
            frameDetectionsArray[i, :] = np.array(splitFrameData[5 * i:5 * i + 5])
        return frameDetectionsArray

    def getNumFrames(self):
        return len(self.data)

def intersectOverUnion(outputDetections, groundTruthObject):
    gtArea = (groundTruthObject[2]-groundTruthObject[0]) * (groundTruthObject[3]-groundTruthObject[1])
    detArea = 0.0
    intersection = 0.0
    for outputDet in outputDetections:
        overlap = getOverlap(outputDet[1:], groundTruthObject)
        if overlap > intersection:
            intersection = overlap
            detArea = (outputDet[3]-outputDet[1]) * (outputDet[4]-outputDet[2])

    return intersection/(gtArea + detArea - intersection)


def calcAcc(vidName, modName):

    modelOut = outputData("./output/modelResults/results_" + os.path.splitext(modName)[0] + "_" +
                          os.path.splitext(vidName)[0] + ".txt")
    numFrames = modelOut.getNumFrames()

    groundTruthLocation = "./groundTruthData/gt_" + os.path.splitext(vidName)[0][:10]
    frameAcc = 0.0
    gtObjectsProcessed = 0
    overNdx = False
    for filename in os.listdir(groundTruthLocation):
        f = os.path.join(groundTruthLocation, filename)
        if os.path.isfile(f) and f.endswith(".txt"):
            fOpen = open(f, 'r')
            gtFrameLines = fOpen.read().splitlines()
            fOpen.close()
            gtFrameList = [line.split(" ") for line in gtFrameLines]

            for gtObj in gtFrameList:
                id = gtObj[0]

                frameNum = int(gtObj[1])
                if frameNum >= numFrames:
                    overNdx = True
                    break
                gtDetections = np.array([int(num) for num in gtObj[2:]])
                outputDetections = modelOut.getFrameData(frameNum)
                if outputDetections.shape[0] > 0:
                    frameAcc += intersectOverUnion(outputDetections, gtDetections)
                gtObjectsProcessed += 1
            if overNdx:
                break
    aveAcc = frameAcc / gtObjectsProcessed

    return aveAcc




if __name__ == '__main__':

    modelName = 'best_yolov8_custom_dataset.pt'
    videoName = "video_0004.mp4"

    print(calcAcc(videoName, modelName))
