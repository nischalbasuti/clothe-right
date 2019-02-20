#!/usr/bin/env python
import os
import glob

import numpy as np
import cv2
import scipy.io as sio

import time
import copy

import pymeanshift as pms

class ImageProcessed(object):
    """docstring for ImageProcessed"""

    def __init__(self, image, annotation):
        """
        image: numpy matrix (BGR image)
        annotation: numpy matrix (Grayscale image)
        """
        super(ImageProcessed, self).__init__()
        self.image      =  image
        self.annotation = annotation

    def display_image(self, prefix="", **kwargs):
        window_name = "%simage" % prefix
        cv2.namedWindow(window_name, 0)
        cv2.imshow(window_name, self.image)

    def display_annotation(self, prefix="", **kwargs):
        window_name = "%annotation" % prefix
        cv2.namedWindow(window_name, 0)
        cv2.imshow(window_name, self.annotation)

    def display_segmented_image(self, prefix="", **kwargs):
        window_name = "%ssegmented image" % prefix
        cv2.namedWindow(window_name, 0)
        cv2.imshow(window_name, self.segmented_image)

    def display_segmented_annotation(self, prefix="", **kwargs):

        annotation_image = copy.deepcopy(self.image)
        for i in range(len(self.segments)):
            if self.segments[i]["is_annotation"] == True:
                for col in self.segments[i]["points"].keys():
                    rows = self.segments[i]["points"][col]
                    for row in rows:
                        annotation_image[col, row] = (0, 255, 0)
        cv2.namedWindow("%ssegmented annotation image" % prefix, 0)
        cv2.imshow("%ssegmented annotation image" % prefix, annotation_image)

    def display(self, prefix="", **kwargs):
        self.display_image(prefix)
        self.display_annotation(prefix)

    def _segment(self, threshold=0):
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
        self.label_annotation_segments()

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
#################################################################################


