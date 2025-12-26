#yolo
from ultralytics import YOLO
import cv2
# Load pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # 'n' = nano (fast, small)
# Read image
img = cv2.imread(r'C:\Users\nagur\Desktop\cv lab\yolo.jpeg')
# Run detection
results = model.predict(img)
# Show results
results[0].show()
