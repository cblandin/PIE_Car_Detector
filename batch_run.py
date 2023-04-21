from object_predict_Kalman import runCustomModel
from overlayGroundTruth import overlayGT
from calculate_accuracy import calcAcc
from benchmark import benchmark


if __name__ == '__main__':


    modelNames = ['best_yolov8_custom_dataset.pt']
    videoNames = ["video_0003.mp4"]
    performBenchmark = False
    accuracy = []

    for videoName in videoNames:
        for modelName in modelNames:
            # runCustomModel(videoName, modelName)
            # overlayGT(videoName, modelName)
            accuracy.append(calcAcc(videoName, modelName))

        if performBenchmark:
            benchmarkModel = "yolov8n.pt"
            benchmark(videoName, benchmarkModel)
            accuracy.append(calcAcc(videoName, "benchmark_" + benchmarkModel))
            print("Processed " + videoName + " with " + modelName)

    for videoName in videoNames:
        for modelName in modelNames:
            print(videoName + " - " + modelName + " Accuracy: " + str(accuracy.pop(0)))

        if performBenchmark:
            print(videoName + " - benchmark Accuracy: " + str(accuracy.pop(0)))