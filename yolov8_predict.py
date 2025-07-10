from ultralytics import YOLO
from PIL import Image

model = YOLO('yolov8n.pt')
image = Image.open("ciguli.png")
sonuc = model.predict(source=image, save=True)