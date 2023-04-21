from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import os

# TOO DO: PUT LOGIC TO BIN OVERLAPPING BOXES TO PREVIOUS FRAMES ID

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    video = "./videos/video_0001.mp4"

    inputVid = cv2.VideoCapture(video)
    r, frame = inputVid.read()
    h, w, _ = frame.shape
    outputVid = cv2.VideoWriter("./outputVideo/video_0002.mp4", cv2.VideoWriter_fourcc(*'MP4V'),
                                int(inputVid.get(cv2.CAP_PROP_FPS)), (w, h))

    confidenceThreshold = 0.25
    overlapThreshold = 50

    startTime = datetime.now()

    results = model(frame, classes=2, device=0, stream=False)[0]
    resultsList = results.boxes.data.tolist()
    objIDs = -np.ones(len(resultsList))
    for i in np.arange(len(resultsList)):
        result = resultsList[i]
        x1, y1, x2, y2, confidence, class_id = result
        if confidence > confidenceThreshold:
            objIDs[i] = i

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, "CAR: " + str(i), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    outputVid.write(frame)
    r, frame = inputVid.read()
    framesToProcess = 9000
    for i in np.arange(framesToProcess):
        oldIDs = objIDs
        oldResultList = resultsList
        results = model(frame, classes=2, device=0,stream=False)[0]
        resultsList = results.boxes.data.tolist()
        objIDs = -np.ones(len(resultsList))
        maxOverLapResult = np.zeros(len(resultsList))

        print("Frame :" + str(i+1) + " of " + str(framesToProcess))
        for j in np.arange(len(oldResultList)):
            if oldIDs[j] > -1:
                oldX1, oldY1, oldX2, oldY2, _, _ = oldResultList[j]

                maxOverLap = 0
                potentialID = -1
                for k in np.arange(len(resultsList)):
                    x1, y1, x2, y2, confidence, class_id = resultsList[k]
                    overlap = np.max([(np.min([x2, oldX2]) - np.max([x1, oldX1])),0]) * \
                              np.max([(np.min([y2,oldY2]) - np.max([y1, oldY1])),0])
                    if overlap > maxOverLap and resultsList[k][4] > confidenceThreshold:
                        maxOverLap = overlap
                        potentialID = k
                if maxOverLap > overlapThreshold and potentialID > -1 and maxOverLapResult[potentialID] < maxOverLap:
                    maxOverLapResult[potentialID] = maxOverLap
                    objIDs[potentialID] = oldIDs[j]
        for j in np.arange(len(resultsList)):
            if resultsList[j][4] > confidenceThreshold:
                if objIDs[j] < 0:
                    objIDs[j] = np.max(objIDs) + 1

                cv2.rectangle(frame, (int(resultsList[j][0]), int(resultsList[j][1])),
                              (int(resultsList[j][2]), int(resultsList[j][3])), (0, 255, 0), 4)
                cv2.putText(frame, "CAR: " + str(int(objIDs[j])),
                            (int(resultsList[j][0]), int(resultsList[j][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, "Frame: " + str(i),
                    (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        outputVid.write(frame)
        r, frame = inputVid.read()
    elapsedTime = datetime.now() - startTime
    print("Elapsed Time: " + str(elapsedTime.seconds))
    inputVid.release()
    outputVid.release()
    cv2.destroyAllWindows()