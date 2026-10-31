
from ultralytics import YOLO
from pathlib import Path

# Original trained classification model
model_path = "models/best.pt"

# Load the trained model
print("Loading YOLO model...")
model = YOLO(model_path)

# Check model classes
print("Model classes:", model.names)

# Export to ONNX
print("Exporting model to ONNX...")

onnx_path = model.export(
    format="onnx",
    imgsz=224,
    opset=12,
    simplify=False,
    dynamic=False,
    half=False
)

print("\nExport completed!")
print("ONNX model saved at:", onnx_path)
