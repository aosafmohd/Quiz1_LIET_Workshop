##### ***Face Detection and Feature Localization***

##### 

This project detects faces in an image, localizes the nose tip and the centers of the eyes, and annotates the image with bounding boxes and labels. The implementation uses MediaPipe Face Mesh and OpenCV.



###### ***Steps to Run the Code***



Clone or copy the project files

Save the Python script (e.g., face\_landmark\_detection.py) and place your test image in a known folder.



Install dependencies

Make sure you have Python 3.8+ installed, then install required packages:



pip install opencv-python mediapipe numpy





Update image path

In the script, set your input image path:



input\_image = r"C:\\path\\to\\your\\testImage.jpeg"





Run the program

From terminal or VS Code:



python face\_landmark\_detection.py





View results



The annotated image will be displayed in a pop-up window.



Press any key (if using cv2.waitKey(0)) or press q (if using the loop) to close.



The program also saves the annotated image (default: annotated\_output.jpg) in the same folder.



###### ***Dependencies Required***



Python 3.8+



OpenCV

&nbsp;(cv2) – for image processing and display



MediaPipe

&nbsp;– for face mesh and landmark detection



NumPy

&nbsp;– for averaging landmark coordinates



Install them with:



pip install opencv-python mediapipe numpy



###### ***Assumptions Made***



Single image input: The program processes one image at a time (not live webcam/video).



Clear face visibility: Works best when the face is clearly visible and not heavily occluded.



Landmark indices: Nose tip = landmark 1, eye centers computed from a set of predefined landmark indices (based on MediaPipe Face Mesh).



Multiple faces: The program supports up to 5 faces per image (max\_num\_faces=5). Each face gets annotated separately.



Windows/Mac/Linux support: Works on any OS where OpenCV can open GUI windows. If GUI windows fail, the annotated image is still saved to disk

