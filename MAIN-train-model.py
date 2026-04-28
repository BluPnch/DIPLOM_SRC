from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")
    model.train(
        data="dataset/data.yaml",
        epochs=50,
        batch=4,     
        imgsz=416,  
        device=0,
        single_cls=True,
        workers=0,
        amp=False
    )