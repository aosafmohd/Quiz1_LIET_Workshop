import cv2
import numpy as np
import os

# ---------- SETTINGS ----------
INPUT_VIDEO = "People_walking.mp4"   # your input video
OUTPUT_VIDEO = "People_walking_blurred.mp4"  # output file
CONF_THRESHOLD = 0.5                 # face detection threshold
# -------------------------------

# Load model files (must exist in the same folder as script)
PROTO_TXT = "deploy.prototxt"
CAFFE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

# Load DNN face detector
net = cv2.dnn.readNetFromCaffe(PROTO_TXT, CAFFE_MODEL)

# Open video
cap = cv2.VideoCapture("People_walking.mp4")
if not cap.isOpened():
    print("Error: cannot open video file")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # codec
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

def blur_face(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    k = max(15, (x2 - x1) // 3 | 1)   # kernel size, odd number
    blurred = cv2.GaussianBlur(roi, (k, k), 30)
    frame[y1:y2, x1:x2] = blurred

while True:
    ret, frame = cap.read()
    if not ret:
        break  # video finished

    # Detect faces
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    h, w = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Keep box within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            blur_face(frame, x1, y1, x2, y2)

    # Show live preview
    cv2.imshow("Blurred Faces", frame)

    # Save frame to output video
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Done! Blurred video saved as {OUTPUT_VIDEO}")
import cv2
import os

INPUT_VIDEO = r"C:\Users\Aosaf\Videos\People_walking.mp4"

print("Does file exist?", os.path.exists(INPUT_VIDEO))

cap = cv2.VideoCapture(INPUT_VIDEO)
print("Video opened:", cap.isOpened())

cap.release()
