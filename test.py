from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO('yolov8m.pt')
    model.train(data="D:\Python/traking/yolov8/data.yaml", epochs=100, imgsz=640)
