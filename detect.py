#!/usr/bin/env python
import os
import argparse

import numpy as np
import cv2

import patched_cnn as pcnn
import image_processor as ip

import copy

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="path to image.")
parser.add_argument("--model", help="path to trained model h5 file.")
parser.add_argument("--shape", help="shape of input")
args = parser.parse_args()

image_source = args.image
model_source = args.model
shape = int(args.shape)

img = cv2.imread(image_source)

imgLab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

input_img = np.array([ cv2.resize( imgLab, (shape, shape) ) ]) # input to nn should be (1, 32, 32, 3)

cnn = pcnn.Patched_CNN()
cnn.load_model(model_source)
print("...initialized model...")

pred = cnn.predict(input_img)

annotation_pred = cv2.resize( pred.reshape(shape, shape), (img.shape[1], img.shape[0]) )

imgp = ip.ImageProcessed(img, annotation_pred)
imgp._segment()

imgp.display_segmented_annotation()
imgp.display_segmented_image()

cv2.waitKey(0)

