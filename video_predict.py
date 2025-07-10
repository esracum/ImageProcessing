from ultralytics import YOLO
import cv2
import os

# YOLOv8 modelini yükle
model = YOLO("yolov8n.pt")

# Giriş videosu
video_path = "ciguli.mp4"
cap = cv2.VideoCapture(video_path)

# Video çıkışı ayarları
save_output = True
output_path = "runs/detect/predict/output_predict.mp4"
os.makedirs("runs/detect/predict", exist_ok=True)

# Sabit video çözünürlüğü
target_width = 640
target_height = 480

fps = int(cap.get(cv2.CAP_PROP_FPS))

# VideoWriter başlat
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

# Kare kare işle
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Boyutlandır
    frame = cv2.resize(frame, (target_width, target_height))

    # YOLOv8 tahmini yap
    results = model(frame, imgsz=640)

    # Tahminleri görselleştir
    annotated_frame = results[0].plot()

    # Görüntüyü ekranda göster
    cv2.imshow("YOLOv8 Video Detection", annotated_frame)

    # Çıktıya yaz
    if save_output:
        out.write(annotated_frame)

    # 'q' ile çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
