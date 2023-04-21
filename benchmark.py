from ultralytics import YOLO
import os
import cv2
import numpy as np
from datetime import datetime
from filterpy.kalman import KalmanFilter
from project_common import getCentroid, getOverlap, getBoundingBox, notEnclosed, createKalmanFilter




def benchmark(videoName, detectorWeights="yolov8n.pt"):
    model = YOLO(detectorWeights)
    if detectorWeights == "best_yolov8_custom_dataset.pt":
        cls = [0]
    else:
        cls = [2]
    results = model.track('./videos/' + videoName, classes=cls, device=0, stream=True, conf=.25, iou=.7)
    resultsFile = open("./output/modelResults/results_benchmark" + "_" + os.path.splitext(detectorWeights)[0] + "_" +
                       os.path.splitext(videoName)[0] + ".txt", 'w')

    for result in results:
        firstIDWritten = False
        for detection in result.boxes.data.tolist():
            if len(detection) == 7:
                x1, y1, x2, y2, id, _, _ = detection
            else:
                x1, y1, x2, y2, _, _ = detection
                id = 0
            if firstIDWritten:
                resultsFile.write(", ")
            firstIDWritten = True
            resultsFile.write(str(int(id)) + ", " + str(int(x1)) + ", " + str(int(y1)) + ", " +
                              str(int(x2)) + ", " + str(int(y2)))
        resultsFile.write("\n")

    resultsFile.close()
    print(videoName + " with " + detectorWeights + " completed")





if __name__ == '__main__':
    detector = 'yolov8n.pt'
    videoDirectory = "./videos"
    allVid = True
    videos = ["video_0001.mp4"]

    if allVid:
        for filename in os.listdir(videoDirectory):
            f = os.path.join(videoDirectory, filename)
            if os.path.isfile(f) and f.endswith(".mp4"):
                benchmark(detector, os.path.splitext(filename)[0])
    else:
        for videoName in videos:
            f = os.path.join(videoDirectory, videoName)
            if os.path.isfile(f) and f.endswith(".mp4"):
                benchmark(detector, os.path.splitext(videoName)[0])
