#!/usr/bin/env python
import os
import argparse

import numpy as np
import cv2

import patched_cnn as pcnn
import image_processor as ip

import copy

def generate_heatmap_image(image, annotation):
    heat_map = copy.deepcopy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            heat_map[i,j, 1] = annotation[i, j] * 255
    return heat_map

def detect(image_path, model_path):
    image_path = args.image
    model_path = args.model
    shape = int(args.shape)

    imgp = ip.ImageProcessed(cv2.imread(image_path))
    img = imgp.get_person_images()[0]["image"]

    imgLab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    input_img = np.array([ cv2.resize( imgLab, (shape, shape) ) ]) # input to nn should be (1, 32, 32, 3)

    cnn = pcnn.Patched_CNN()
    cnn.load_model(model_path)
    print("...initialized model...")

    pred = cnn.predict(input_img)

    annotation_pred = cv2.resize( pred.reshape(shape, shape), (img.shape[1], img.shape[0]) )

    imgp.image = img
    imgp.annotation = annotation_pred

    imgp._segment()

    imgp.display_segmented_annotation()
    imgp.display_segmented_image()

    return imgp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", help="path to image.")
    parser.add_argument("--model", "-m", help="path to trained model h5 file.")
    parser.add_argument("--shape", "-s", help="shape of input")
    args = parser.parse_args()

    imgp_detected = detect(args.image, args.model)

    imgp_detected.display_image()
    imgp_detected.display_annotation()
    imgp_detected.display_segmented_annotation()
    imgp_detected.display_segmented_image()

    heat_map = generate_heatmap_image(imgp_detected.image, imgp_detected.annotation)
    cv2.namedWindow("heat map", 0)
    cv2.imshow("heat map", heat_map)

    cv2.waitKey(0)

