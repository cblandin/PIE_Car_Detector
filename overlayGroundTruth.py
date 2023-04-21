import cv2
import numpy as np
import os

def overlayGT(vidName, modName):
    groundTruthLocation = "./output/groundTruthData/gt_" + os.path.splitext(vidName)[0][:10]
    gtList = []
    frameList = []
    for filename in os.listdir(groundTruthLocation):
        f = os.path.join(groundTruthLocation, filename)
        if os.path.isfile(f) and f.endswith(".txt"):
            fOpen = open(f, 'r')
            gtFrameLines = fOpen.read().splitlines()
            fOpen.close()
            gtFrameList = [line.split(" ") for line in gtFrameLines]
            frameList.append(int(gtFrameList[0][1]))
            gtList.append(gtFrameList)

    inputVid = cv2.VideoCapture("./output/modelOutputVideos/output_" + os.path.splitext(modName)[0] + "_" + vidName)
    r, frame = inputVid.read()
    h, w, _ = frame.shape
    outputVid = cv2.VideoWriter("./output/groundTruthOverlayVideos/gtVideo_" + os.path.splitext(modName)[0] + "_" + vidName
                                , cv2.VideoWriter_fourcc(*'MP4V'), int(inputVid.get(cv2.CAP_PROP_FPS)), (w, h))

    curFrame = 0
    while r:
        if curFrame in frameList:
            ndx = frameList.index(curFrame)
            for gtObj in gtList[ndx]:
                _, _, x1, y1, x2, y2 = gtObj
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
            outputVid.write(frame)
        r, frame = inputVid.read()
        curFrame += 1

    outputVid.release()
    inputVid.release()
    cv2.destroyAllWindows()
    print("./output/groundTruthOverlay/gtVideo_" + os.path.splitext(modName)[0] + "_" + vidName + " saved")




if __name__ == '__main__':

    modelName = 'best_yolov8_custom_dataset.pt'
    videoName = "video_0004.mp4"
    overlayGT(modelName, videoName)


