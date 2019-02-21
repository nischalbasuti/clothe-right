#!/usr/bin/env python
import os
import glob

import numpy as np
import cv2
import scipy.io as sio

from clothing_co_parsing_helpers import label_dict, tops_dict
from image_processor import ImageProcessed

def process_mat_files(image_dir_path, annotation_dir_path,
                      save_dir='./clean_data', **kwargs):
    """
    This function processes and saves shit.
    """

    # Create directories to save the processed files.
    image_save_dir = os.path.join(save_dir, "image")
    annotation_save_dir = os.path.join(save_dir, "annotation")
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(annotation_save_dir, exist_ok=True)

    # Process images.
    image_paths = glob.glob(os.path.join(image_dir_path, "*.jpg"))
    for image_path in image_paths:
        # open image
        print(image_path)
        image = cv2.imread(image_path) 

        # open annotation file
        image_file_name = os.path.basename(image_path)
        annotation_file_name = image_file_name[:-3] + "mat"
        annotation_path = os.path.join(annotation_dir_path, annotation_file_name)
        print(annotation_path)

        annotation = sio.loadmat(annotation_path)['groundtruth']

        # Get annotation of only tops.
        tops_annotation = np.zeros(annotation.shape)
        print(tops_dict.values())
        for i in range(annotation.shape[0]):
            for j in range(annotation.shape[1]):
                tops_annotation[i,j] = 255 if annotation[i,j] in tops_dict.values() else 0

        # Save image and annotation.
        image_save_path = os.path.join(image_save_dir, image_file_name)
        annotation_save_path = os.path.join(annotation_save_dir, image_file_name)
        cv2.imwrite(image_save_path, image)
        cv2.imwrite(annotation_save_path, tops_annotation)

def process_clean_files(image_dir_path="./clean_data/image",
        annotation_dir_path="./clean_data/annotation",
        save_dir='./cropped_data', **kwargs):
    """
    This function processes and saves shit.
    """

    # Create directories to save the processed files.
    image_save_dir = os.path.join(save_dir, "image")
    annotation_save_dir = os.path.join(save_dir, "annotation")
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(annotation_save_dir, exist_ok=True)

    # Process images.
    image_paths = glob.glob(os.path.join(image_dir_path, "*.jpg"))
    for image_path in image_paths:
        # open image
        print(image_path)
        image = cv2.imread(image_path) 

        # open annotation file
        image_file_name = os.path.basename(image_path)
        annotation_path = os.path.join(annotation_dir_path, image_file_name)
        print(annotation_path)

        annotation = cv2.imread(annotation_path)

        ip = ImageProcessed(image, annotation)
        # TODO loop through all person images?
        cropped_image, cropped_annt = ip.get_person_images()[0].values()

        # Save image and annotation.
        image_save_path = os.path.join(image_save_dir, image_file_name)
        annotation_save_path = os.path.join(annotation_save_dir, image_file_name)
        cv2.imwrite(image_save_path, cropped_image)
        cv2.imwrite(annotation_save_path, cropped_annt)

if __name__  == '__main__':
    # process the clothing-co-parsing dataset.
    process_mat_files(image_dir_path="./clothing-co-parsing/photos",
            annotation_dir_path="./clothing-co-parsing/annotations/pixel-level",
            save_dir='./clean_data')

    # process the cleaned data and save cropped images.
    process_clean_files(image_dir_path="./clean_data/image",
            annotation_dir_path="./clean_data/annotation",
            save_dir='./cropped_data')
