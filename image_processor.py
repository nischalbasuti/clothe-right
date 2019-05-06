#!/usr/bin/env python
import os
import glob

import numpy as np
import cv2
import scipy.io as sio

import time
import copy

import pickle

from matplotlib import pyplot as plt

import pymeanshift as pms

class ImageProcessed(object):
    """
    A convenient class to manipulate an image and it's annotation.
    Uses reference code from https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html for YOLO.
    """

    def __init__(self, image, annotation=None):
        """
        image: numpy matrix (BGR image)
        annotation: numpy matrix (Grayscale image)
        """
        super(ImageProcessed, self).__init__()
        self.image      =  image
        self.annotation = annotation if annotation is not None else np.zeros(image.shape)

    def display_image(self, prefix="", matplot=False, **kwargs):
        window_name = "%simage" % prefix
        if matplot:
            plt.imshow(self.image)
            plt.show()
        else:
            cv2.namedWindow(window_name, 0)
            cv2.imshow(window_name, self.image)

    def display_annotation(self, prefix="", matplot=False, **kwargs):
        window_name = "%annotation" % prefix
        if matplot:
            plt.imshow(self.annotation)
            plt.show()
        else:
            cv2.namedWindow(window_name, 0)
            cv2.imshow(window_name, self.annotation)

    def display_segmented_image(self, prefix="", matplot=False, **kwargs):
        window_name = "%ssegmented image" % prefix
        if matplot:
            plt.imshow(self.segmented_image)
            plt.show()
        else:
            cv2.namedWindow(window_name, 0)
            cv2.imshow(window_name, self.segmented_image)

    def display_segmented_annotation(self, prefix="", matplot=False, **kwargs):

        annotation_image = copy.deepcopy(self.image)
        for i in range(len(self.segments)):
            if self.segments[i]["is_annotation"] == True:
                for col in self.segments[i]["points"].keys():
                    rows = self.segments[i]["points"][col]
                    for row in rows:
                        annotation_image[col, row] = (0, 255, 0)
        if matplot:
            plt.imshow(annotation_image)
            plt.show()
        else:
            cv2.namedWindow("%ssegmented annotation image" % prefix, 0)
            cv2.imshow("%ssegmented annotation image" % prefix, annotation_image)

    def display(self, prefix="", **kwargs):
        self.display_image(prefix)
        self.display_annotation(prefix)

    def _segment(self, threshold=0.5):
         ##
         # Segment the image.
        start_time_seconds = time.time()
        print("...started segmenting...")
        # Using mean shift implementation from https://github.com/fjean/pymeanshift

        segmented_image, labels_image, number_regions = pms.segment(
                                                            self.image,
                                                            spatial_radius = 6,
                                                            range_radius = 4.5,
                                                            min_density = 50)
        # segmented_image, labels_image, number_regions = pms.segment(
        #                                                     self.image,
        #                                                     spatial_radius = 1,
        #                                                     range_radius = 1,
        #                                                     min_density = 300)

        # Gather points of each segment.
        self._set_segment_points(labels_image, number_regions)

        # Label each segment if it's annotated or note.
        self.label_annotation_segments(threshold=threshold)

        self.segmented_image = segmented_image

        return segmented_image

    def _set_segment_points(self, labels_image, number_regions):
        """
        Add pixel points of each segment into the list self.segments.
        """

        # points are a dict, with columns as keys and list of rows as values.
        #   segments =  [ 
        #                   {
        #                       "points":   {
        #                                       "col": [ row0, row1,... ]
        #                                   }...,
        #                       "maxPoint": [ minX, minY ], 
        #                       "minPoint": [ maxX, maxY ] 
        #                   } ...  
        #               ]
        self.segments = [ { "points": {},
                            "maxPoint": [0,0],
                            "minPoint": [1000,1000]
                          } for i in range(number_regions)]

        # This loop will:
        # 1. Find the minimum and maximum row and column values.
        # 2. Find the points that are inside the segment.
        for col in range(self.image.shape[0]):
            for row in range(self.image.shape[1]):

                # store the segments in a list.

                # Find the minimum row and col values.
                if self.segments[labels_image[col, row]]["minPoint"][0] > col:
                    self.segments[labels_image[col, row]]["minPoint"][0] = col
                if self.segments[labels_image[col, row]]["minPoint"][1] > row:
                    self.segments[labels_image[col, row]]["minPoint"][1] = row

                # Find the maximum row and col values.
                if self.segments[labels_image[col, row]]["maxPoint"][0] < col:
                    self.segments[labels_image[col, row]]["maxPoint"][0] = col
                if self.segments[labels_image[col, row]]["maxPoint"][1] < row:
                    self.segments[labels_image[col, row]]["maxPoint"][1] = row

                # Initialize the first time the col comes up.
                if col not in self.segments[labels_image[col, row]]['points'].keys():
                    self.segments[labels_image[col, row]]['points'][col] = []

                # Add the row to the corresponding list of rows for col.
                self.segments[labels_image[col, row]]['points'][col].append(row)

    def label_annotation_segments(self, threshold=0.5):
        """
        Labels a segment as a annotation if ratio of annotation points to non-annotation
        points is over the threshold(default 0.5)
        """
        for i in range(len(self.segments)):
            annotation_point_count = 0
            total_point_count = 0
            for col in self.segments[i]["points"].keys():
                rows = self.segments[i]["points"][col]
                for row in rows:
                    total_point_count += 1
                    if self.annotation[col, row] > threshold:
                        annotation_point_count += 1

            if annotation_point_count/total_point_count > threshold:
                # print(annotation_point_count/total_point_count)
                self.segments[i]["is_annotation"] = True
            else:
                self.segments[i]["is_annotation"] = False

    def get_segment_images(self, shape=(32, 32)):
        return [ cv2.resize(segment["image"], shape)  for segment in self.segments ]

    def get_person_bboxes(self, yolo_weights="./yolo_files/yolov3.weights",
                         yolo_config="./yolo_files/yolov3.cfg",
                         yolo_classes="./yolo_files/coco.names",
                         display = False, **kwargs):
        """
        Method to get bounding boxes of self.image.
        
        reference code from:
        https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html
        """
        Width = self.image.shape[1]
        Height = self.image.shape[0]
        scale = 0.00392

        # read class names from text file
        self.classes = None
        with open(yolo_classes, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # generate different colors for different classes 
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # read pre-trained model and config file
        net = cv2.dnn.readNet(yolo_weights, yolo_config)

        # create input blob 
        blob = cv2.dnn.blobFromImage(self.image, scale, (416,416), (0,0,0), True, crop=False)

        # set input blob for the network
        net.setInput(blob)

        # run inference through the network
        # and gather predictions from output layers
        outs = net.forward(self._get_output_layers(net))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # for each detetion from each output layer 
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.99 and class_id == 0:
                    print(confidence)
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        self.bound_image = copy.deepcopy(self.image)
        # go through the detections remaining
        # after nms and draw bounding box
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            
            self._draw_bounding_box(class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        if display:
            # display output image    
            cv2.imshow("object detection", self.bound_image)
            # wait until any key is pressed
            cv2.waitKey()

        return [ {'x': int(b[0]), 'y': int(b[1]), 'w': int(b[2]), 'h': int(b[3])} for b in boxes ]


    def _get_output_layers(self, net):
        """
        Method to get the output layer names in the architecture

        reference code from:
        https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html
        """
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def _draw_bounding_box(self, class_id, confidence, x, y, x_plus_w, y_plus_h):
        """
        Method to draw bounding box on the detected object with class name

        reference code from:
        https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html
        """
        label = str(self.classes[class_id])
        color = self.COLORS[class_id]
        cv2.rectangle(self.bound_image, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(self.bound_image, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def get_person_images(self, display=False):
        """ 
        returns array images and their annotations cropped to the persons
        detected in self.image
        """
        boxes = self.get_person_bboxes()
        person_images = []
        for b in boxes:
            img = self.image[b['y']:b['y']+b['h'], b['x']:b['x']+b['w']]
            annt = self.annotation[b['y']:b['y']+b['h'], b['x']:b['x']+b['w']]

            # A shitty way to filter out some background detections.
            if img.shape[0] <= 100 or img.shape[1] <= 100:
                continue

            pose = self._get_person_pose(img)
            person_images.append({'image': img, 'annotation': annt, 'pose': pose})
            if display:
                cv2.namedWindow("person", 0)
                cv2.imshow("person", img)
                cv2.waitKey()
        return person_images

    def _get_person_pose(self, image, threshold=0.0, show = False, **kwargs):
        """
        reference code from:
        https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/
        """
        # Specify the paths for the 2 files
        protoFile = "./pose_deploy_linevec_faster_4_stages.prototxt.txt"
        weightsFile = "./pose_iter_160000.caffemodel"
         
        # Read the network into Memory
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

        # Specify the input image dimensions
        inWidth = 368
        inHeight = 368
        inWidth = image.shape[1]
        inHeight = image.shape[0]
         
        frame = copy.deepcopy(self.image)
        # Prepare the frame to be fed to the network
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
         
        # Set the prepared object as the input blob of the network
        net.setInput(inpBlob)

        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]

        frameWidth = inWidth
        frameHeight = inHeight
        # Empty list to store the detected keypoints
        points = []
        # pose_mask = np.zeros(self.image.shape)
        pose_mask = np.zeros((image.shape[0], image.shape[1], 15))
        # for i in range(len(output[0])):
        for i in range(15):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
         
            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
             
            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H
         
            if prob > threshold : 
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, lineType=cv2.LINE_AA)
         
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))

                for _y in range(int(y)-5, int(y)+5):
                    for _x in range(int(x)-5, int(x)+5):
                        pose_mask[_y, _x, i] = 255
            else :
                points.append(None)

        POSE_PAIRS = [
                (0,1), (1, 2), (1, 5), (2,3), (3, 4), (5,6), (6, 7), 
                (1,14), (14, 8), (14, 11), (8, 9), (9, 10), (11, 12),
                (12, 13)
                ]

        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
         
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 0), 3)
         
        if show == True:
            cv2.namedWindow("Output-Keypoints", 0)
            cv2.imshow("Output-Keypoints",frame)
            # cv2.namedWindow("pose mask", 0)
            # cv2.imshow("pose mask", pose_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        
        ip = np.dstack((image, pose_mask))
        # with open("arr_dump.pickle", "wb") as f_out:
        #     pickle.dump(ip, f_out)

        # self.pose_mask = pose_mask
        return ip

if __name__ == '__main__':
    for file in os.listdir("./clean_data/image"):
        print(file)
        ip = ImageProcessed(cv2.imread("./clean_data/image/%s" % file),
                cv2.imread("./clean_data/annotation/%s" % file) )
        # imgs = ip.get_person_images(display = True)
        ip.get_person_pose(image=ip.image)
    # ip = ImageProcessed(cv2.imread("./jimih1.jpg"))
    # ip.get_person_pose(show = True)

