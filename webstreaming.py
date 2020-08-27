# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import tensorflow as tf
from imutils.video import FPS
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import dlib
import numpy as np

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

model_path = "models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
threshold = 0.7
odapi = ""

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def detect_motion(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock

    cctvpath = "rtsp://10.1.1.10:554/rtsp_live/profile_token_0"
    cap = cv2.VideoCapture(cctvpath)
    #cap = cv2.VideoCapture('test_videos/v2.mp4')

    W = None
    H = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    totalExit = 0
    totalEnterance = 0

    fps = FPS().start()

    # loop over frames from the video stream
    while True:

        rects = []
        trackers = []

        r, img = cap.read()

        if img is None:
            break

        #img = cv2.resize(img, (1280, 720))
        img = imutils.resize(img, width=1024)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = img.shape[:2]
        
        borderPositionX = W - (W // 4)
        borderWidth = 40

        cv2.rectangle(img, (borderPositionX, 0), (borderPositionX + borderWidth, H), (0, 255, 255), 2)

        boxes, scores, classes, num = odapi.processFrame(img)

        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                cv2.putText(img, 'Person : ' + str(round(scores[i], 2)), (box[1], box[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(box[1], box[0], box[3], box[2])
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)

        for tracker in trackers:
            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            
            start = (startX, startY)
            end = (endX, endY)
            
            color = list(np.random.random(size=3) * 256)
            
            cv2.rectangle(img, start, end, color, 4)
           
            rects.append((startX, startY, endX, endY))

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID

            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                #y directions of the centroids
                y = [c[1] for c in to.centroids]

                #x directions of the centroids
                x = [c[0] for c in to.centroids]
                
                avgPosX = np.mean(x)
                startPositionX = centroid[0]

                direction = centroid[0] - avgPosX
                to.centroids.append(centroid)

                if not to.counted:
                    #if not counted and but re-recognized in the area and difference between intial position and last position
                    #its not between the border, do not increase the number.
                    if (borderPositionX > startPositionX and borderPositionX > avgPosX) or (borderPositionX < startPositionX and borderPositionX < avgPosX):
                        pass
                    else:
                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        if direction < 0 and centroid[0] < borderPositionX:
                            totalEnterance += 1
                            to.counted = True

                        # if the direction is positive (indicating the object
                        # is moving down) AND the centroid is below the
                        # center line, count the object
                        elif direction > 0 and centroid[0] > borderPositionX:
                            totalExit += 1
                            to.counted = True

            text = "ID {}".format(objectID)
            cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(img, (centroid[0] ,centroid[1]), 4, (0, 255, 0), -1)

            trackableObjects[objectID] = to

        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            ("Enter", totalEnterance),
            ("Exit", totalExit),
            ("Person Inside" , str(totalEnterance - totalExit))
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(img, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        fps.update()

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = img.copy()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage.tobytes()) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    #initialize params
    odapi = DetectorAPI(path_to_ckpt=model_path)
 
    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
