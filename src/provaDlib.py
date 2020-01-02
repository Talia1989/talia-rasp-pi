import cv2
import numpy as np
import dlib
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

#init camera
dimpx = 624
dimpy = 480
camera = PiCamera()
camera.rotation = 180
camera.resolution = (dimpx, dimpy)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(dimpx, dimpy))
time.sleep(0.1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/pi/Desktop/repo/talia-rasp-pi/landmarks/shape_predictor_68_face_landmarks.dat")

#init dimensionamento punti
def midpoint(p1,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2) 

for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = f.array
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)
        #riga orizzontale occhio
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        #riga verticale occhio
        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))
        
        #linea orizzontale e verticale:
        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0),2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0),2)

        for n in range(36, 42): #range di punti per disegnare il viso --> 36, 42 occhio destro
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
    rawCapture.truncate(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("Frame", frame)

cv2.destroyAllWindows()