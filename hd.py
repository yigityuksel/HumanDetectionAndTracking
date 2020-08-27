# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import dlib

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

        #print("Elapsed Time:", end_time-start_time)

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

if __name__ == "__main__":
    model_path = 'models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7

    #cctvpath = "rtsp://10.1.1.10:554/rtsp_live/profile_token_0"
    #cap = cv2.VideoCapture(cctvpath)
    cap = cv2.VideoCapture('test_videos/v1.mp4')

    outputPath = "test_v2.avi"
    writer = None

    W = None
    H = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=40, maxDistance=80)
    trackers = []
    trackableObjects = {}

    totalExit = 0
    totalEnterance = 0

    fps = FPS().start()

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
        borderWidth = 0

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if outputPath is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(outputPath, fourcc, 30,
                (W, H), True)

        #cv2.rectangle(img, (borderPositionX, 0), (borderPositionX + borderWidth, H), (0, 255, 255), 2)

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

        #print("objects : " + str(len(objects)))
        #print("rects : " + str(len(rects)))

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID

            to = trackableObjects.get(objectID, None)

            #print("Object is {}".format(objectID))

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

            #print("ID {} - position {} {} ".format(objectID, str(centroid[0]), str(centroid[1])))
            text = "ID {}".format(objectID)
            
            color_ = list(np.random.random(size=3) * 256)

            cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_, 2)
            cv2.circle(img, (centroid[0] ,centroid[1]), 4, color_, -1)

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

                # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(img)


        cv2.imshow("preview", img)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        
        fps.update()
    

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    # stop the timer and display FPS information 
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

