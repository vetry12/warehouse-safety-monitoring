from ultralytics import YOLO

# Load the pre-trained YOLOv8 nano model
# Downloads automatically on first run
model = YOLO('yolov8n.pt')

# Run detection on our test image
results = model('test.jpg')

# Save the result with bounding boxes drawn
results[0].save(filename='result.jpg')

# Print summary
print("Done! Check result.jpg to see the detections.")
print(f"Detected {len(results[0].boxes)} objects.")