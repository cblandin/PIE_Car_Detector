import numpy as np
from filterpy.kalman import KalmanFilter


def getCentroid(result):
    x1, y1, x2, y2,_,_ = result
    return np.array([[(x1 + x2)/2], [(y1 + y2)/2]])

def createKalmanFilter(result, dt):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.vstack((getCentroid(result), np.zeros((2, 1))))
    kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kf.P *= 5
    kf.R *= 2.5
    kf.Q = np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])*30
    return kf

def getBoundingBox(result, x):

    x1, y1, x2, y2, _, _ = result
    width = x2 - x1
    height = y2-y1

    newX1 = x[0] - width/2.0
    newX2 = x[0] + width/2.0
    newY1 = x[1] - height/2.0
    newY2 = x[1] + height/2.0

    return [newX1, newY1, newX2, newY2]

def getOverlap(box1, box2):
    b1x1, b1y1, b1x2, b1y2 = box1
    b2x1, b2y1, b2x2, b2y2 = box2
    return np.max([(np.min([b1x2, b2x2]) - np.max([b1x1, b2x1])), 0]) * \
                  np.max([(np.min([b1y2, b2y2]) - np.max([b1y1, b2y1])), 0])

def notEnclosed(resultsList, i, threshold=.85):
    x1, y1, x2, y2, _, _ = resultsList[i]
    area = (y2-y1)*(x2-x1)
    box1 = resultsList[i][0:4]

    for j in np.arange(len(resultsList)):
        box2 = resultsList[j][0:4]
        overlap = getOverlap(box1, box2)
        if (i != j and overlap/area > threshold):
            return False

    return True