
from ultralytics import YOLO
from pathlib import Path

# Test image
image_path = r"C:\Users\rsarm\OneDrive\Desktop\Project AgroFarmer\test_images\Diease unknown.jpg"

# Load both models
print("Loading original model...")
pt_model = YOLO("models/best.pt")

print("Loading ONNX model...")
onnx_model = YOLO("models/best.onnx")

# Run prediction using original model
print("\nOriginal PyTorch prediction:")
pt_results = pt_model.predict(image_path, verbose=False)

pt_probs = pt_results[0].probs
pt_class = pt_probs.top1
pt_confidence = float(pt_probs.top1conf)

print("Class:", pt_model.names[pt_class])
print("Confidence:", round(pt_confidence, 4))

# Run prediction using ONNX model
print("\nONNX prediction:")
onnx_results = onnx_model.predict(image_path, verbose=False)

onnx_probs = onnx_results[0].probs
onnx_class = onnx_probs.top1
onnx_confidence = float(onnx_probs.top1conf)

print("Class:", onnx_model.names[onnx_class])
print("Confidence:", round(onnx_confidence, 4))

# Compare results
print("\nComparison:")

if pt_class == onnx_class:
    print("SUCCESS: Both models predicted the same class.")
else:
    print("WARNING: Predictions are different.")
