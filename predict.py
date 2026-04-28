from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model.predict(
    source="test.jpg",
    save=True,              # автоматически сохраняет результат в папку runs/predict/
    conf=0.25,              # можно поднять, чтобы отсеять ложные срабатывания
)

print("Результат сохранён в runs/predict/")