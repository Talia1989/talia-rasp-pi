import cv2
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

# init part
dimpx = 624
dimpy = 420
camera = PiCamera()
# camera.rotation = 180
camera.resolution = (dimpx, dimpy)
camera.framerate = 32
camera.exposure_compensation = 25
rawCapture = PiRGBArray(camera, size=(dimpx, dimpy))
time.sleep(0.1)
face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/repo/talia-rasp-pi/haar/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/pi/Desktop/repo/talia-rasp-pi/haar/haarcascade_righteye_2splits.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)


def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    return frame


def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
#         eyecenter = x + w / 2  # get the eye center
#         if eyecenter < width * 0.5:
#             left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    return left_eye, right_eye

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)

    return img


def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    print(keypoints)
    return keypoints


def nothing(x):
    pass


def main():
#     camera.capture(rawCapture, format="bgr")
    
    
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
#         _, frame = cap.read()
        frame = f.array
        face_frame = detect_faces(frame, face_cascade)
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)
            print(eyes)
            if eyes is not None:
                for eye in eyes:
                    if eye is not None:
                        threshold = r = cv2.getTrackbarPos('threshold', 'image')
                        print("threshold is {} ".format(threshold))
                        eye = cut_eyebrows(eye)
                        keypoints = blob_process(eye, threshold, detector)
                        eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('image', frame)
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#     cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()