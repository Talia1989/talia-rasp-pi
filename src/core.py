import cv2 #import openCV
import os #funzioni di libreria per accedere al sistema operativo
import numpy as np #import numpy: funzioni matematiche, grafici..

def faceDetection(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converte l'immagine a colore in scala di grigi
    classificatore=cv2.CascadeClassifier(r"/home/pi/repo/talia-rasp-pi/haar/haarcascade_frontalface_default.xml") #algoritmo classificatore(astrazione matematica -- immagini in matrici)
    face= classificatore.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=6)
    return face,gray #faccia in grigio

#cicla il contenuto della cartella alla ricerca di immagini
def creaDataset(directory): 
    faces=[]
    facesID=[]
    i: int
    i=0
    for path,subdir,files in os.walk(directory):
        for filename in files:
            print("TRAINING IMAGE ", i)
            i+=1
            if filename.startswith("."):
                print("skipping system file")
                continue
            id=os.path.basename(path) #basename di id: "0" oppure "1"
            image_path=os.path.join(path,filename) #image_path:prende il percorso completo dell'immagine      path: percorso fino alle cartelle "0" e "1"
            img_test=cv2.imread(image_path) #img_test: converte l'immagine in una matrice secondo la transcodifica di opencv
            if img_test is None:
                print("error opening image")
                continue
            face,gray=faceDetection(img_test) #ritorna l'immagine convertita i scala di grigi 
            if len(face) != 1:
                print(len(face))
                continue
            (x,y,w,h)=face[0]
            region=gray[y:y+w,x:x+h]
            faces.append(region)
            facesID.append(int(id))
    return faces,facesID


def addestramento(faces,facesID):
    face_recognizer=cv2.face.createLBPHFaceRecognizer()
    face_recognizer.train(faces,np.array(facesID))
    return face_recognizer

def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

def put_name(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.QT_FONT_NORMAL,2,(0,0,255),3)
    
    
################################# modalità esecuzione   #########################################

def init_addestramento():
    print("inizio addestramento del modello!")
    faces,facesID=creaDataset('/home/pi/repo/talia-rasp-pi/immagini')
    faceRecognizer=addestramento(faces,facesID)
    faceRecognizer.save("modello/trainedData.yml") #trainedData: dataset addestrato (astrazione matematica) --> modello

def carica_addestramento():
    faceRecognizer = cv2.face.createLBPHFaceRecognizer()
    faceRecognizer.load(r"/home/pi/repo/talia-rasp-pi/src/modello/trainedData.yml")
    return faceRecognizer
    
#################################  inizio video capture   #############################################################
init_addestramento()
faceRecognizer = carica_addestramento()

name = {1:"Talia", 2:"Andrea"}

capture = cv2.VideoCapture(0) #cattura il video da webcam

while True:
    ret, test_img=capture.read()
    faces_detected,gray=faceDetection(test_img)

    for face in faces_detected:
        (x,y,w,h)=face
        region=gray[y:y+w,x:x+h]
        label,confidence = faceRecognizer.predict(region)
        print("label", label)
        print("confidence",confidence)
        predictedName=name[label]
        draw_rect(test_img,face)
        if confidence>75:
            put_name(test_img,"?",x,y)
            continue
        else: put_name(test_img,predictedName,x,y)
    resized = cv2.resize(test_img, (800, 800))

    cv2.imshow("face", resized)
    if cv2.waitKey(10) == 27:
        break

#################################  fine video capture  #############################################################



