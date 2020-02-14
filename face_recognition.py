import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def detect_face(img):

    # kernel = np.array([[-1,-1,-1], 
    #             [-1, 9,-1],
    #             [-1,-1,-1]])

    # sharpened = cv2.filter2D(img, -1, kernel)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascPath = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascPath)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    return faces,gray

def labels_for_training_data(directory):
    faces=[]
    faceID=[]
    i=-1
    name_dict = {}

    for path,subdirnames,filenames in os.walk(directory):
        i+=1
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping ystem file")
                continue
            name_dict[i]=os.path.basename(path)
            id=i
            img_path=os.path.join(path,filename)
            print("img_path",img_path)
            print("id: ",id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("Image not read properly")
                continue

            faces_rect,gray_img=detect_face(test_img)
            
            if len(faces_rect)!=1:
                continue
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))
            print("faceID : ",faceID)
    return faces,faceID,name_dict

def train_classifier(faces,faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(40,40,225),thickness=2)

def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x-3,y-10),cv2.FONT_HERSHEY_DUPLEX,1,(44,44,225),2)

