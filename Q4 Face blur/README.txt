Steps to Run the Code

Clone or download this project folder.
Place your input video (e.g., People_walking.mp4) in the same folder as the script, or provide the full path.

Download face detection model files (must be in the same folder as the script):

deploy.prototxt

res10_300x300_ssd_iter_140000.caffemodel

Install dependencies:
pip install opencv-python opencv-contrib-python numpy

Run the script:
python face_blur_video.py

The script will:

Open the input video (People_walking.mp4 by default).

Detect and blur faces frame by frame.

Save the processed video as People_walking_blurred.mp4.

Display a live preview (press q to exit early).

Dependencies

Python 3.8+

OpenCV (opencv-python, opencv-contrib-python)

NumPy

Optional:

FFmpeg (recommended if OpenCV has trouble reading MP4 files)

Assumptions

The input video is named People_walking.mp4 and located in the same directory as the script, unless the full path is specified in the code.

OpenCV’s DNN face detector (deploy.prototxt + res10_300x300_ssd_iter_140000.caffemodel) is available in the working directory.

Output is saved using the mp4v codec. If playback fails, convert to AVI format (XVID codec) or use FFmpeg to re-encode.