from object_predict_Kalman import runCustomModel
from overlayGroundTruth import overlayGT
from calculate_accuracy import calcAcc
from benchmark import benchmark


if __name__ == '__main__':


    modelName = 'best_yolov8_custom_dataset.pt'
    videoName = "video_0001_1min.mp4"
    # runCustomModel(videoName, modelName)
    overlayGT(videoName, modelName)
    print(calcAcc(videoName, modelName))


    # benchmarkModel = "yolov8n.pt"
    # benchmark(videoName, benchmarkModel)
    # print(calcAcc(videoName, "benchmark_" + benchmarkModel))