from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('best.pt')
    results = model.train(data='vehicle.yaml', epochs=10, batch=16, device=0)
