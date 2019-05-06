#!/usr/bin/env python
import os
import glob
import pickle
import json

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
        try:
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
        except FileNotFoundError:
            print("file %s not found." % image_path)


def process_clean_files(image_dir_path="./clean_data/image",
        annotation_dir_path="./clean_data/annotation",
        save_dir='./cropped_data', **kwargs):
    """
    This function processes and saves shit.
    """

    # Create directories to save the processed files.
    image_save_dir = os.path.join(save_dir, "image")
    os.makedirs(image_save_dir, exist_ok=True)

    annotation_save_dir = os.path.join(save_dir, "annotation")
    os.makedirs(annotation_save_dir, exist_ok=True)

    pose_save_dir = os.path.join(save_dir, "pose")
    os.makedirs(pose_save_dir, exist_ok=True)

    dataset = { 'image': [], 'annotation': [], 'pose_heatmap': [] }

    # Process images.
    count = 0
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

        # if count < 852:
        #     continue

        ip = ImageProcessed(image, annotation)
        
        prev_area = 0
        for person in ip.get_person_images():
            cropped_image, cropped_annt, cropped_pose = person.values()
            
            # Only save the largest person detected in image.
            area = cropped_image.shape[0] * cropped_image.shape[1]
            if area < prev_area:
                continue

            # if prev_area == 0:
            #     dataset['image'].append(cropped_image)
            #     dataset['annotation'].append(cropped_annt)
            #     dataset['pose_heatmap'].append(cropped_pose)
            # elif area > prev_area:
            #     dataset['image'][-1]        = cropped_image
            #     dataset['annotation'][-1]   = cropped_annt
            #     dataset['pose_heatmap'][-1] = cropped_pose

            prev_area = area

            # Save image and annotation.
            image_save_path = os.path.join(image_save_dir, image_file_name)
            annotation_save_path = os.path.join(annotation_save_dir, image_file_name)
            pose_save_path = os.path.join(pose_save_dir, image_file_name)

            cv2.imwrite(image_save_path, cropped_image)
            cv2.imwrite(annotation_save_path, cropped_annt)

            # Save pose heatmap.
            cv2.imwrite(pose_save_path+".0.jpg", cropped_pose[:,:, 0:3])
            cv2.imwrite(pose_save_path+".1.jpg", cropped_pose[:,:, 3:6])
            cv2.imwrite(pose_save_path+".2.jpg", cropped_pose[:,:, 6:9])
            cv2.imwrite(pose_save_path+".3.jpg", cropped_pose[:,:, 9:12])
            cv2.imwrite(pose_save_path+".4.jpg", cropped_pose[:,:, 12:15])
            cv2.imwrite(pose_save_path+".5.jpg", cropped_pose[:,:, 15:18])

            print(pose_save_path+".0.jpg", cropped_pose[:,:, 0:3].shape)
            print(pose_save_path+".1.jpg", cropped_pose[:,:, 3:6].shape)
            print(pose_save_path+".2.jpg", cropped_pose[:,:, 6:9].shape)
            print(pose_save_path+".3.jpg", cropped_pose[:,:, 9:12].shape)
            print(pose_save_path+".4.jpg", cropped_pose[:,:, 12:15].shape)
            print(pose_save_path+".5.jpg", cropped_pose[:,:, 15:18].shape)
            
            # with open(pose_save_path, 'wb') as f:
                # print(cropped_pose.shape)
                # pickle.dump(cropped_pose, f)
        # if count % 10 == 0:
        #     with open("./dataset.pickle", "wb") as f:
        #         print("Writing dataset.pickle")
        #         print("Size:", len(dataset['image']))
        #         pickle.dump(dataset, f)
            # with open("./dataset.json", "w") as f:
            #     print("Writing dataset.json")
            #     print("Size:", len(dataset['image']))
            #     json.dump(dataset, f)
        count += 1

    # with open("./dataset.pickle", "wb") as f:
    #     print("Writing dataset.pickle")
    #     print("Size:", len(dataset))
    #     pickle.dump(dataset, f)
    # with open("./dataset.json", "w") as f:
    #     print("Writing dataset.json")
    #     print("Size:", len(dataset['image']))
    #     json.dump(dataset, f)

if __name__  == '__main__':
    # # process the clothing-co-parsing dataset.
    # process_mat_files(image_dir_path="./clothing-co-parsing/photos",
    #         annotation_dir_path="./clothing-co-parsing/annotations/pixel-level",
    #         save_dir='./clean_data')

    # process the cleaned data and save cropped images.
    process_clean_files(image_dir_path="./clean_data/image",
            annotation_dir_path="./clean_data/annotation",
            save_dir='./cropped_data')
