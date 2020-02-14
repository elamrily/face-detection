import cv2
import os
import sys
import numpy as np
import argparse
import face_recognition as fr 
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
from generate_dataset import detect
from flask import request,Flask,render_template,jsonify,Response
from flask_bootstrap import Bootstrap
import os
import shutil
import sys

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str, default="kcf",
    help="OpenCV object tracker type")
args = vars(ap.parse_args())
OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# Initialise the visage detector
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

camera_port = 0
camera_width = 1080
camera_height = 800

#%% Run web site:
app=Flask(__name__)

Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('blockchain.html')
    
@app.route('/track_visualisation',methods=['POST'])
def track_visualisation():

    # initialize the bounding box coordinates of the object we are going to track
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
    initBB = (229, 112, 184, 234)
    param_track = True
    video_capture = cv2.VideoCapture(camera_port) # + cv2.CAP_DSHOW)
    
    # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,camera_width)
    # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,camera_height)

    ret, test_img = video_capture.read()
    tracker.init(test_img, initBB)
    # initialize the FPS throughput estimator
    fps = FPS().start()
    # id of img
    img_id = 0
    tmp = 0
    state = 0
    while state == 0:
        # Capture frame-by-frame
        ret, test_img = video_capture.read()
        
        if test_img is None:
            print('None')
            video_capture = cv2.VideoCapture(camera_port) # + cv2.CAP_DSHOW)
    
            # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,camera_width)
            # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,camera_height)
            ret, test_img = video_capture.read()
            print(test_img)
            tracker.init(test_img, initBB)
            fps = FPS().start()
            tmp = 0

        if test_img is not None:
            
            # grab the new bounding box coordinates of the object
            if tmp == 1:
                box = initBB
                success = True
            else:
                (success, box) = tracker.update(test_img)
            
            print(success)
            print(box)

            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(test_img, (x, y), (x + w, y + h),
                    (0, 255, 0), 2)
                # face cropping only
                new_img = test_img[y:y+h,x:x+w]
                img = detect(new_img,faceCascade, img_id)
                img_id += 1
            tmp += 1    
        
            # update the FPS counter
            fps.update()
            fps.stop()

            ret, jpeg = cv2.imencode('.jpg', test_img)  
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/select_face',methods=['POST'])
def select_face():

    video_capture = cv2.VideoCapture(camera_port) # + cv2.CAP_DSHOW)
    # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,camera_width)
    # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,camera_height)

    state = 0
    while state == 0:
        # Capture frame-by-frame
        ret, test_img = video_capture.read()
        if test_img is None:
            # print(test_img)
            video_capture = cv2.VideoCapture(camera_port) # + cv2.CAP_DSHOW)
            # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,camera_width)
            # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,camera_height)

            ret, test_img = video_capture.read()

        if test_img is not None:
            faces_detected,gray_img = fr.detect_face(test_img)        
            initBB = (229, 112, 184, 234)
            cv2.rectangle(test_img, (229, 112), (229 + 184, 112 + 234),(40, 40, 225), 2)

            # c = cv2.selectROI(test_img)
            # print(c)
            

            ret, jpeg = cv2.imencode('.jpg', test_img)  

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/stop',methods=['POST'])
def stop():

    video_capture = cv2.VideoCapture(camera_port) # + cv2.CAP_DSHOW)    
    # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,camera_width)
    # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,camera_height)

    state = 0
    while state == 0:
        # Capture frame-by-frame
        ret, test_img = video_capture.read()
        if test_img is None:
            video_capture = cv2.VideoCapture(camera_port) # + cv2.CAP_DSHOW)
    
            # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,camera_width)
            # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,camera_height)
            ret, test_img = video_capture.read()

        if test_img is not None:
            faces_detected,gray_img = fr.detect_face(test_img)        
            ret, jpeg = cv2.imencode('.jpg', test_img)  

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/start',methods=['POST'])
def start():
    # initialize the bounding box coordinates of the object we are going to track

    # initialize the FPS throughput estimator
    fps = None
    img_id = 0

    video_capture = cv2.VideoCapture(camera_port) # + cv2.CAP_DSHOW)
    # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,camera_width)
    # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,camera_height)

    faces,faceID,name_dict=fr.labels_for_training_data('../face_reco_app/dataset')
    face_recognizer=fr.train_classifier(faces,faceID)
    param_track = False

    state = 0
    while state == 0:
        # Capture frame-by-frame
        ret, test_img = video_capture.read()

        if test_img is None:
            video_capture = cv2.VideoCapture(camera_port) # + cv2.CAP_DSHOW)
            # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,camera_width)
            # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,camera_height)

            ret, test_img = video_capture.read()

        if test_img is not None:
            faces_detected,gray_img = fr.detect_face(test_img)
            key = cv2.waitKey(1) & 0xFF

            # face detect par
            if param_track == False :
                for face in faces_detected:
                    (x,y,w,h)=face
                    roi_gray=gray_img[y:y+h,x:x+w]
                    label,confidence=face_recognizer.predict(roi_gray)
                    print("\nconfidence:",confidence)
                    print("label     :",label)
                    if confidence < 55:
                        fr.draw_rect(test_img,face)
                        predicted_name=name_dict[label]
                        fr.put_text(test_img,predicted_name,x,y)
                    else:
                        fr.draw_rect(test_img,face)
                        predicted_name="John Doe"
                        fr.put_text(test_img,predicted_name,x,y)
            
            ret, jpeg = cv2.imencode('.jpg', test_img)

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/restart_visualisation',methods=['POST'])
def restart_visualisation():

    # initialize the bounding box coordinates of the object we are going to track
    initBB = None
    # initialize the FPS throughput estimator
    fps = None
    # id of img
    img_id = 0

    video_capture = cv2.VideoCapture(camera_port) # + cv2.CAP_DSHOW)
    # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,camera_width)
    # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,camera_height)

    faces,faceID,name_dict=fr.labels_for_training_data('../face_reco_app/dataset')
    face_recognizer=fr.train_classifier(faces,faceID)
    param_track = False

    state = 0
    while state == 0:
        # Capture frame-by-frame
        ret, test_img = video_capture.read()

        if test_img is None:
            video_capture = cv2.VideoCapture(camera_port) # + cv2.CAP_DSHOW)
            # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,camera_width)
            # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,camera_height)
            
            ret, test_img = video_capture.read()

        if test_img is not None:
            faces_detected,gray_img = fr.detect_face(test_img)
            key = cv2.waitKey(1) & 0xFF

            # face detect par
            if param_track == False :
                for face in faces_detected:
                    (x,y,w,h)=face
                    roi_gray=gray_img[y:y+h,x:x+w]
                    label,confidence=face_recognizer.predict(roi_gray)
                    print("\nconfidence:",confidence)
                    print("label     :",label)
                    if confidence < 100:
                        fr.draw_rect(test_img,face)
                        predicted_name=name_dict[label]
                        fr.put_text(test_img,predicted_name,x,y)
                    else:
                        fr.draw_rect(test_img,face)
                        predicted_name="John Doe"
                        fr.put_text(test_img,predicted_name,x,y)
            
            ret, jpeg = cv2.imencode('.jpg', test_img)  

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/rename_folder',methods=['POST'])
def rename_folder():
    base = '../face_reco_app/dataset/'
    text = request.form['name']
    new_name = text.upper()
    list_person = os.listdir(base)
    
    if 'new_face' in list_person:
        if new_name in list_person:
            shutil.rmtree(base + new_name)
            os.rename((base + 'new_face'),(base + new_name))
        else:
            os.rename((base + 'new_face'),(base + new_name))
    
    list_person = os.listdir(base)

    return 1

@app.route('/clear_database',methods=['POST'])
def clean_database():
    
    base = '../face_reco_app/dataset/'
    
    for element in os.listdir(base):
        print(element)
        if str(element) != "John Doe":
            shutil.rmtree(base + str(element))

@app.route('/video_feed_track')
def video_feed_track():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return(Response(track_visualisation(), mimetype='multipart/x-mixed-replace; boundary=frame'))

@app.route('/video_feed_restart')
def video_feed_restart():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return(Response(restart_visualisation(), mimetype='multipart/x-mixed-replace; boundary=frame'))

@app.route('/video_feed_select')
def video_feed_select():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return(Response(select_face(), mimetype='multipart/x-mixed-replace; boundary=frame'))

@app.route('/video_feed_start')
def video_feed_start():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return(Response(start(), mimetype='multipart/x-mixed-replace; boundary=frame'))

@app.route('/video_feed_stop')
def video_feed_stop():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return(Response(stop(), mimetype='multipart/x-mixed-replace; boundary=frame'))


if __name__=='__main__':
    app.run(debug=True,port='1234')


# %%
