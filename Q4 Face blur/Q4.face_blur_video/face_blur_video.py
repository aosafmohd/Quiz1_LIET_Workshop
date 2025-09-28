import cv2
import numpy as np

# ---- Settings ----
VIDEO_PATH = "People_walking.mp4"    # replace with your video file
CONF_THRESHOLD = 0.5
DISPLAY_WIDTH = 800         # resize for faster display

# Load OpenCV DNN model
proto = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto, model)

def blur_face(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    # Kernel size depends on face size
    k = max(15, (x2 - x1) // 3 | 1)   # must be odd
    blurred = cv2.GaussianBlur(roi, (k, k), 30)
    frame[y1:y2, x1:x2] = blurred

# Open video file
cap = cv2.VideoCapture("People_walking.mp4")

if not cap.isOpened():
    print("Error: cannot open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    # Resize for faster processing
    h, w = frame.shape[:2]
    scale = DISPLAY_WIDTH / w
    small = cv2.resize(frame, (DISPLAY_WIDTH, int(h * scale)))

    # Detect faces
    blob = cv2.dnn.blobFromImage(small, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([small.shape[1],
                                                       small.shape[0],
                                                       small.shape[1],
                                                       small.shape[0]])
            (x1, y1, x2, y2) = box.astype("int")

            # Scale back to original frame size
            x1 = int(x1 / scale); y1 = int(y1 / scale)
            x2 = int(x2 / scale); y2 = int(y2 / scale)

            # Apply blur
            blur_face(frame, x1, y1, x2, y2)

    cv2.imshow("Blurred Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
