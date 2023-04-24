from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('best.pt')
    inputs = '01148.png'
    results = model(inputs, save=True, device=0)

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs

print(boxes)