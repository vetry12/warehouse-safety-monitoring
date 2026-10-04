from ultralytics import YOLO
import cv2
import time

# Load the YOLOv8 nano model
model = YOLO('yolov8n.pt')

# Open the input video
video_path = 'test_video.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video info: {width}x{height} @ {fps} FPS, {total_frames} frames total")

# Set up the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# Process frame by frame
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection on this frame
    results = model(frame, verbose=False)

    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Save frame to output video
    out.write(annotated_frame)

    frame_count += 1

    # Print progress every 30 frames
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed
        print(f"Processed {frame_count}/{total_frames} frames | {current_fps:.1f} FPS")

# Clean up
cap.release()
out.release()

# Final stats
total_time = time.time() - start_time
avg_fps = frame_count / total_time
print(f"\n Done!")
print(f"Processed {frame_count} frames in {total_time:.1f} seconds")
print(f"Average FPS: {avg_fps:.1f}")
print(f"Output saved as: output_video.mp4")