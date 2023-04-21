from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
from filterpy.kalman import KalmanFilter
from project_common import getCentroid, getOverlap, getBoundingBox, notEnclosed, createKalmanFilter
import os


def runCustomModel(vidName, modName):
    model = YOLO(modName)
    if modName == 'best_yolov8_custom_dataset.pt':
        cls = [0]
    else:
        cls = [2]

    video = "./videos/" + vidName
    resultsFile = open("./output/modelResults/results_" + os.path.splitext(modName)[0]+ "_" + os.path.splitext(vidName)[0] + ".txt", 'w')

    inputVid = cv2.VideoCapture(video)
    r, frame = inputVid.read()
    h, w, _ = frame.shape
    outputVid = cv2.VideoWriter("./output/modelOutputVideos/output_" + os.path.splitext(modName)[0] + "_" + vidName,
                                cv2.VideoWriter_fourcc(*'MP4V'),
                                int(inputVid.get(cv2.CAP_PROP_FPS)), (w, h))

    confidenceThreshold = 0.40
    overlapThreshold = 50

    startTime = datetime.now()

    results = model(frame, classes=cls, device=0, stream=False)[0]
    resultsList = results.boxes.data.tolist()
    objIDs = -np.ones(len(resultsList))
    kalmanFilterList = []

    dt = 1.0 / 30.0
    firstIDWritten = False
    for i in np.arange(len(resultsList)):
        result = resultsList[i]

        kalmanFilterList.append(createKalmanFilter(result, dt))
        x1, y1, x2, y2, confidence, class_id = result
        if confidence > confidenceThreshold:
            objIDs[i] = i
            if notEnclosed(resultsList, i):
                if firstIDWritten:
                    resultsFile.write(", ")
                firstIDWritten = True
                resultsFile.write(str(int(objIDs[i])) + ", " + str(int(x1)) + ", " + str(int(y1)) + ", " +
                                  str(int(x2)) + ", " + str(int(y2)))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, "Veh: " + str(i), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    resultsFile.write("\n")
    outputVid.write(frame)
    r, frame = inputVid.read()
    framesToProcess = int(inputVid.get((cv2.CAP_PROP_FRAME_COUNT)))-1
    for i in np.arange(framesToProcess):
        if not r:
            break
        oldIDs = objIDs
        oldResultList = resultsList

        results = model(frame, classes=cls, device=0, stream=False)[0]
        resultsList = results.boxes.data.tolist()
        objIDs = -np.ones(len(resultsList))
        maxOverLapResult = np.zeros(len(resultsList))

        firstIDWritten = False
        print("Frame :" + str(i + 1) + " of " + str(framesToProcess))
        for j in np.arange(len(oldResultList)):
            if oldIDs[j] > -1:
                kalmanFilterList[j].predict()
                oldX1, oldY1, oldX2, oldY2 = getBoundingBox(oldResultList[j], kalmanFilterList[j].x)
                maxOverLap = 0
                potentialID = -1
                for k in np.arange(len(resultsList)):
                    x1, y1, x2, y2, confidence, class_id = resultsList[k]
                    overlap = getOverlap(resultsList[k][0:4], [oldX1, oldY1, oldX2, oldY2])
                    if overlap > maxOverLap and resultsList[k][4] > confidenceThreshold:
                        maxOverLap = overlap
                        potentialID = k
                if maxOverLap > overlapThreshold and potentialID > -1 and maxOverLapResult[potentialID] < maxOverLap:
                    maxOverLapResult[potentialID] = maxOverLap
                    objIDs[potentialID] = oldIDs[j]
        newKalmanFilterList = []
        for j in np.arange(len(resultsList)):
            if resultsList[j][4] > confidenceThreshold:
                if objIDs[j] < 0:
                    if np.min(objIDs) > 0:
                        objIDs[j] = np.min(objIDs) - 1
                    else:
                        objIDs[j] = np.max(objIDs) + 1
                    newKalmanFilterList.append(createKalmanFilter(resultsList[j], dt))

                else:

                    kalmanFilterList[np.argmax(objIDs[j] == oldIDs)].update(getCentroid(resultsList[j]))
                    newKalmanFilterList.append(kalmanFilterList[np.argmax(objIDs[j] == oldIDs)])

                if notEnclosed(resultsList, j):

                    if firstIDWritten:
                        resultsFile.write(", ")
                    firstIDWritten = True
                    resultsFile.write(str(int(objIDs[j])) + ", " + str(int(x1)) + ", " + str(int(y1)) + ", " + str(
                        int(x2)) + ", " + str(int(y2)))
                    x1, y1, x2, y2 = getBoundingBox(resultsList[j], newKalmanFilterList[j].x)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    cv2.putText(frame, "Veh: " + str(int(objIDs[j])) + ", vx: " + str(
                        np.round(newKalmanFilterList[j].x[2, 0], 1)) + ", vy: " + str(
                        np.round(newKalmanFilterList[j].x[3, 0], 1)), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                newKalmanFilterList.append(KalmanFilter(dim_x=4, dim_z=2))

        resultsFile.write("\n")
        kalmanFilterList = newKalmanFilterList
        cv2.putText(frame, "Frame: " + str(i),
                    (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3, cv2.LINE_AA)
        outputVid.write(frame)
        r, frame = inputVid.read()
    elapsedTime = datetime.now() - startTime
    print("Elapsed Time: " + str(elapsedTime.seconds))
    inputVid.release()
    outputVid.release()
    cv2.destroyAllWindows()
    resultsFile.close()
    print("./output/modelOutputVideos/output_" + os.path.splitext(modName)[0] + "_" + vidName + " saved")
    print("./output/modelResults/results_" + os.path.splitext(vidName)[0] + "_" + os.path.splitext(modName)[0] + ".txt saved")


if __name__ == '__main__':

    modelName = 'best_yolov8_custom_dataset.pt'
    videoName = "video_0001_1min.mp4"
    runCustomModel(modelName, videoName)



