import cv2
import numpy as np
import dlib
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
from math import hypot

#init camera
dimpx = 624
dimpy = 480
camera = PiCamera()
camera.rotation = 180
camera.resolution = (dimpx, dimpy)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(dimpx, dimpy))
time.sleep(0.1)

font = cv2.FONT_HERSHEY_DUPLEX

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/pi/Desktop/repo/talia-rasp-pi/landmarks/shape_predictor_68_face_landmarks.dat")

def nothing(x):
    pass

#init trackbar
windowName = "eye-tracker"
ratioName = "ratio"
cv2.namedWindow(windowName)
cv2.createTrackbar(ratioName, windowName, 0, 10, nothing)

#init dimensionamento punti
def midpoint(p1,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eyeType, landmarks):
    nordEst = 0
    nordOvest = 0
    sudEst = 0
    sudOvest = 0
    est = 0
    ovest = 0
    if eyeType == "L":
        ovest = 36
        est = 39
        nordOvest = 37
        nordEst = 38
        sudEst = 40
        sudOvest = 41
    elif eyeType == "R":
        ovest = 42
        est = 45
        nordOvest = 43
        nordEst = 44
        sudEst = 46
        sudOvest = 47
            
    #riga orizzontale occhio
    left_point = (landmarks.part(ovest).x, landmarks.part(ovest).y)
    right_point = (landmarks.part(est).x, landmarks.part(est).y)
    #riga verticale occhio
    center_top = midpoint(landmarks.part(nordOvest), landmarks.part(nordEst))
    center_bottom = midpoint(landmarks.part(sudOvest), landmarks.part(sudEst))
    
    #linea orizzontale e verticale:
    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0),2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0),2)
    
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    print("lunghezza linea orizzontale: {}, lunghezza linea verticale: {}" .format(hor_line_lenght, ver_line_lenght))
    
    ratioBlink = hor_line_lenght/ver_line_lenght
    return ratioBlink

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
        ratioBlink = get_blinking_ratio("L", landmarks)
        
        maxRatio = r = cv2.getTrackbarPos(ratioName, windowName)
        print("valore maxRatio: {}, valore ratio blink: {}".format(maxRatio, ratioBlink))
        
        if ratioBlink > maxRatio:
            cv2.putText(frame, "BLINK", (50, 150), font, 7, (255, 0 ,0))

        for n in range(36, 42): #range di punti per disegnare il viso --> 36, 42 occhio destro
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
    rawCapture.truncate(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow(windowName, frame)

cv2.destroyAllWindows()