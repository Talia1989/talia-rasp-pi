from flask import Flask
from flask_sse import sse
from flask import Response
from flask_cors import CORS

import cv2
import numpy as np
import dlib
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
from math import hypot



app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
    return "Eye tracking"
 
# def get_message():
# #'''this could be any function that blocks until data is ready'''
#     time.sleep(1.0)
#     s = time.ctime(time.time())
#     print(s)
#     return s

#init camera
dimpx = 624
dimpy = 480
camera = PiCamera()
# camera.rotation = 180
camera.resolution = (dimpx, dimpy)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(dimpx, dimpy))
time.sleep(0.1)

font = cv2.FONT_HERSHEY_SIMPLEX

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/pi/Desktop/repo/talia-rasp-pi/landmarks/shape_predictor_68_face_landmarks.dat")

def nothing(x):
    pass

#init trackbar
windowName = "eye-tracker"
ratioName = "ratio"
thresholdName = "threshold"
gazeMinName = "gazeMin"
gazeMaxName = "gazeMax"
cv2.namedWindow(windowName)
cv2.createTrackbar(ratioName, windowName, 0, 10, nothing)
cv2.createTrackbar(thresholdName, windowName, 0, 255, nothing)
cv2.createTrackbar(gazeMinName, windowName, 0, 2000, nothing)
cv2.createTrackbar(gazeMaxName, windowName, 0, 2000, nothing)

#init dimensionamento punti
def midpoint(p1,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eyeType, landmarks):
    nordEst ,nordOvest, sudEst, sudOvest, est, ovest = (0,)*6
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
#     hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0),2)
#     ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0),2)
    
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
#     print("lunghezza linea orizzontale: {}, lunghezza linea verticale: {}" .format(hor_line_lenght, ver_line_lenght))
    
    ratioS = hor_line_lenght/ver_line_lenght
    return ratioS

def get_threshold(eye, threshold_value):
    eye = cv2.medianBlur(eye, 3)
    _, the_threshold = cv2.threshold(eye, threshold_value, 255, cv2.THRESH_BINARY) #######pupilla nera, occhio bianco
    if the_threshold is not None:
        cv2.imshow("THRESHOLD", cv2.resize(the_threshold, (dimpx, dimpy), fx=5, fy=5))
    return the_threshold

def get_gaze_ratio(eyeType, landmarks, frame, gray, threshold_value):
    nordEst ,nordOvest, sudEst, sudOvest, est, ovest, occhio = (0,)*7
    if eyeType == "L":
        ovest = 36
        est = 39
        nordOvest = 37
        nordEst = 38
        sudEst = 40
        sudOvest = 41
        occhio = "sinistro"
    elif eyeType == "R":
        ovest = 42
        est = 45
        nordOvest = 43
        nordEst = 44
        sudEst = 46
        sudOvest = 47
        occhio = "destro"
        
        
    eye_region = np.array([(landmarks.part(ovest).x, landmarks.part(ovest).y),
                                    (landmarks.part(nordOvest).x, landmarks.part(nordOvest).y),
                                    (landmarks.part(nordEst).x, landmarks.part(nordEst).y),
                                    (landmarks.part(est).x, landmarks.part(est).y),
                                    (landmarks.part(sudEst).x, landmarks.part(sudEst).y),
                                    (landmarks.part(sudOvest).x, landmarks.part(sudOvest).y)], np.int32)
#         cv2.polylines(frame, [eye_region], True, (0, 0, 255), 2) ################################## contorno occhio colore rosso

    #maschera immagine --> mostra solo occhio
    height, width,_ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    the_eye = cv2.bitwise_and(gray, gray, mask = mask)
    
    #ritaglio eye
    min_x = np.min(eye_region[:,0])
    max_x = np.max(eye_region[:,0])
    min_y = np.min(eye_region[:,1])
    max_y = np.max(eye_region[:,1])
    
    eye = the_eye[min_y: max_y, min_x: max_x]
    
    #estrazione threshold
    threshold_eye = get_threshold(eye, threshold_value)
    gaze_ratio = -1
    if threshold_eye is not None:
        h_t, w_t = threshold_eye.shape
        left_side_threshold = threshold_eye[0: h_t, 0: int(w_t/2)]
        right_side_threshold = threshold_eye[0: h_t, int(w_t/2): w_t]
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_white = cv2.countNonZero(right_side_threshold)
        if left_side_white == 0:
            gaze_ratio = 0
        elif right_side_white == 0:
            gaze_ratio = h_t*w_t
        else:
            gaze_ratio = left_side_white/right_side_white
    return gaze_ratio
            
            

@app.route('/stream')
def stream():
    def eventStream():
#         while True:
#             # wait for source data to be available, then push it
#             yield 'data: {}\n\n'.format(get_message())
        for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            print("--------------------------------------------------------------------------------------------------------")
            frame = f.array
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = detector(gray)
            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()

                landmarks = predictor(gray, face)
                
                #Detect blinking
                ratioS = get_blinking_ratio("L", landmarks)
                ratioD = get_blinking_ratio("R", landmarks)
                
                maxRatio = r = cv2.getTrackbarPos(ratioName, windowName)
                threshold_value = t = cv2.getTrackbarPos(thresholdName, windowName)
                gaze_ratio_min = gmin = cv2.getTrackbarPos(gazeMinName, windowName)
                gaze_ratio_max = gmax = cv2.getTrackbarPos(gazeMaxName, windowName)
                
                if ratioS > maxRatio and ratioD > maxRatio: 
                    cv2.putText(frame, "BLINK", (50, 150), font, 5, (0, 255 ,0)) 
                
                #Gaze detection
                left_eye_gaze = get_gaze_ratio("L", landmarks, frame, gray, threshold_value)
                right_eye_gaze = get_gaze_ratio("R", landmarks, frame, gray, threshold_value)
                print("valore gaze sinistro:{}, valore gaze destro:{}".format(left_eye_gaze, right_eye_gaze))
                
                gaze_ratio_tot = int(((left_eye_gaze + right_eye_gaze)/2)*1000)
                
                messageResponse = ""
                if gaze_ratio_tot <= gaze_ratio_min:
                    cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255),3)
                    messageResponse = "RIGHT"
                elif gaze_ratio_min < gaze_ratio_tot < gaze_ratio_max:
                    cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255),3)
                    messageResponse = "CENTER"
                else:
                    cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255),3)
                    messageResponse = "LEFT"
                
                
                cv2.putText(frame, str(gaze_ratio_tot), (50,300), font, 2, (0,0,255),3)
                
                yield 'data: {}\n\n'.format(messageResponse)
                
            rawCapture.truncate(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            cv2.imshow(windowName, frame)

        cv2.destroyAllWindows()
    return Response(eventStream(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run()
