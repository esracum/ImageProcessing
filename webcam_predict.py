from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt") 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 ile tahmin
    results = model(frame, imgsz=640)

    # Sonucu çizdir
    annotated_frame = results[0].plot()

    # Ekrana göster
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
