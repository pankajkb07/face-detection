import cv2

face_cascade = cv2.CascadeClassifier('C:/Users/panka/AppData/Roaming/Python/Python311/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/panka/AppData/Roaming/Python/Python311/site-packages/cv2/data/haarcascade_eye.xml')

video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = col[y:y+h, x:x+w]
        roi_color = video_data[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
    cv2.imshow('LIVE FACE DETECTION', video_data)
    
    if cv2.waitKey(10) == ord('a'):
        break

video_cap.release()
cv2.destroyAllWindows()




'''


# with explaination 

import cv2  # import the OpenCV library

# create cascade classifiers for detecting faces and eyes
face_cap = cv2.CascadeClassifier('C:/Users/panka/AppData/Roaming/Python/Python311/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/panka/AppData/Roaming/Python/Python311/site-packages/cv2/data/haarcascade_eye.xml')

# open the default camera device
video_cap = cv2.VideoCapture(0)

# start an infinite loop to read frames from the camera and perform face detection
while True:
    # read a frame from the camera
    ret, video_data = video_cap.read()

    # convert the frame to grayscale for faster processing
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # display the frame with the detected faces
    cv2.imshow('LIVE FACE DETECTION', video_data)

    # exit the loop if the 'a' key is pressed
    if cv2.waitKey(10) == ord('a'):
        break

# release the camera device and destroy all windows
video_cap.release()
cv2.destroyAllWindows()



'''