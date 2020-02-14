import cv2
import os 
import numpy as np


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def generate_dataset(img, id, img_id):
    path = os.path.join("../face_reco_app/dataset/new_face")
    makedirs(path)
    cv2.imwrite(os.path.join(path,str(id)+str(img_id)+".jpg"), img)

    print('new face write in folder')
    return

def draw_boundary(img, classifier, scaleFactor, minNeighbors):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    if len(features) > 0:
        print('one face')
        for (x,y,w,h) in features:
            #cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            #cv2.putText(img, text, (x,y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            coords = [x,y,w,h]
            print(coords)
    else :
        print('no face')
        coords = []
    return(coords)
    

def detect(img, faceCascade, img_id):
    color = (255,0,0)
    coords = draw_boundary(img, faceCascade, 1.3, 5)
    
    if len(coords)==4:
        roi_img = img[coords[1]:(coords[1]+coords[3]) , coords[0]:(coords[0]+coords[2])]
        user_id = 1
        generate_dataset(roi_img, user_id, img_id)
    return(img)
