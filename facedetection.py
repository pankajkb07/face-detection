# live face detection in python 

import cv2

face_cap = cv2.CascadeClassifier('C:/Users/panka/AppData/Roaming/Python/Python311/site-packages/cv2/data/haarcascade_frontalface_default.xml')

video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('LIVE  FACE DETECTION', video_data)
    
    if cv2.waitKey(10) == ord('a'):
        break

video_cap.release()
cv2.destroyAllWindows()


"""
This code imports the OpenCV library using the "cv2" alias. 
It then loads a pre-trained cascade classifier (haarcascade_frontalface_default.xml) 
that is used to detect faces in images or video streams. 


The code sets up a video capture object using the first available camera (indexed at 0).
It then runs a loop to capture each frame from the camera and apply face detection to it. 


If a face is detected, a rectangle is drawn around it in green. The frame with the rectangle
is then displayed in a window titled "FACE DETECTION". 


The loop continues until the user presses the "a" key. After that, 
the video capture is released and all windows are destroyed.

"""