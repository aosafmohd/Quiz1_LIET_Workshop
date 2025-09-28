"""
face_blur_save_clip.py
- Press 'r' to toggle ongoing recording (start/stop)
- Press 's' to save the last N seconds (retroactive clip)
- Press 'q' to quit
"""
import cv2
import numpy as np
import time
import os
from collections import deque
import urllib.request
import imutils

# ---- Settings ----
CONF_THRESHOLD = 0.5
BUFFER_SECONDS = 8          # how many seconds to keep in the circular buffer
OUTPUT_DIR = "clips"
USE_DNN = True             # try DNN first, fallback to Haar cascade
SAVE_FOURCC = 'mp4v'       # try 'mp4v' or 'XVID' if mp4 doesn't work
CAMERA_SRC = 0             # 0 for default webcam; can use "rtsp://..." for CCTV
DISPLAY_WIDTH = 800        # window width (resized for faster processing)

# DNN model files (auto-download if missing)
PROTO_TXT = "deploy.prototxt"
CAFFE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/res10_300x300_ssd_iter_140000.caffemodel"

def maybe_download(url, dest):
    if os.path.exists(dest):
        return
    print(f"Downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")

# small helper to pixelate a region
def pixelate(roi, blocks=10):
    (h, w) = roi.shape[:2]
    if h == 0 or w == 0:
        return roi
    x = max(1, w // blocks)
    y = max(1, h // blocks)
    temp = cv2.resize(roi, (x, y), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

# choose blur kernel that is odd and proportional to face size
def auto_kernel(w, h):
    k = max(3, int(min(w, h) / 3))
    if k % 2 == 0:
        k += 1
    return k

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try to prepare DNN model
net = None
if USE_DNN:
    try:
        maybe_download(PROTO_URL, PROTO_TXT)
        maybe_download(MODEL_URL, CAFFE_MODEL)
        net = cv2.dnn.readNetFromCaffe(PROTO_TXT, CAFFE_MODEL)
        print("Loaded DNN face detector.")
    except Exception as e:
        print("DNN detector not available:", e)
        net = None

# Haar cascade fallback
cascade = None
if net is None:
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        print("Using Haar cascade for face detection.")
    except Exception as e:
        print("No face detector available:", e)
        raise SystemExit("Cannot continue without any detector.")

# Open camera/stream
cap = cv2.VideoCapture(CAMERA_SRC)
if not cap.isOpened():
    raise SystemExit("Failed to open video source. Check CAMERA_SRC.")

# Try to read FPS and frame size
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 20.0   # fallback
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
print(f"Source FPS={fps:.1f} W={width} H={height}")

# buffer to keep last N seconds of processed frames
buffer_size = max(1, int(fps * BUFFER_SECONDS))
frame_buffer = deque(maxlen=buffer_size)

recording = False
writer = None

def make_writer(filename, fourcc=SAVE_FOURCC):
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    return cv2.VideoWriter(filename, fourcc_code, fps, (width, height))

print("Controls: r=toggle recording, s=save last {}s, q=quit".format(BUFFER_SECONDS))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received. Exiting.")
        break

    # resize for speed but keep original for saving (we'll map boxes back)
    small = imutils.resize(frame, width=DISPLAY_WIDTH)
    (h_s, w_s) = small.shape[:2]
    scale_x = frame.shape[1] / float(w_s)
    scale_y = frame.shape[0] / float(h_s)

    faces = []
    if net is not None:
        blob = cv2.dnn.blobFromImage(small, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < CONF_THRESHOLD:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w_s, h_s, w_s, h_s])
            (x1, y1, x2, y2) = box.astype("int")
            # scale back to original frame coords
            x1o = max(0, int(x1 * scale_x))
            y1o = max(0, int(y1 * scale_y))
            x2o = min(frame.shape[1], int(x2 * scale_x))
            y2o = min(frame.shape[0], int(y2 * scale_y))
            faces.append((x1o, y1o, x2o, y2o, confidence))
    else:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        for (x, y, w, h) in rects:
            x1o = max(0, int(x * scale_x))
            y1o = max(0, int(y * scale_y))
            x2o = min(frame.shape[1], int((x + w) * scale_x))
            y2o = min(frame.shape[0], int((y + h) * scale_y))
            faces.append((x1o, y1o, x2o, y2o, None))

    # Anonymize faces
    for (x1, y1, x2, y2, conf) in faces:
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        roi = frame[y1:y2, x1:x2]

        # Choose method: blur OR pixelate — change here if you prefer
        # 1) Blur
        k = auto_kernel(w, h)
        try:
            blurred = cv2.GaussianBlur(roi, (k, k), 0)
        except Exception:
            blurred = roi

        # 2) Optionally pixelate instead:
        # blurred = pixelate(roi, blocks=16)

        frame[y1:y2, x1:x2] = blurred

        # draw a thin rectangle (optional)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        if conf is not None:
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

    # UI overlays
    status = "REC" if recording else "IDLE"
    cv2.putText(frame, f"[r] Record toggle  [s] Save last {BUFFER_SECONDS}s  [q] Quit    Status:{status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # push processed frame to buffer (so saved clips include anonymized faces)
    frame_buffer.append(frame.copy())

    # if recording, write frame to writer
    if recording and writer is not None:
        writer.write(frame)

    cv2.imshow("Face Blurring (press q to quit)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        recording = not recording
        if recording:
            fname = os.path.join(OUTPUT_DIR, f"recording_{int(time.time())}.mp4")
            writer = make_writer(fname)
            print("Started recording ->", fname)
        else:
            if writer:
                writer.release()
                print("Stopped recording.")
            writer = None
    elif key == ord('s'):
        # save the last BUFFER_SECONDS from the buffer
        if len(frame_buffer) == 0:
            print("No frames in buffer yet.")
        else:
            fname = os.path.join(OUTPUT_DIR, f"clip_{int(time.time())}.mp4")
            vw = make_writer(fname)
            print(f"Saving last {len(frame_buffer)} frames to {fname} ...")
            for f in frame_buffer:
                vw.write(f)
            vw.release()
            print("Saved:", fname)

# cleanup
if writer:
    writer.release()
cap.release()
cv2.destroyAllWindows()
