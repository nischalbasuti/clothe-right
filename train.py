#!/usr/bin/env python3

import argparse
from patched_cnn import _train
  
parser = argparse.ArgumentParser()
parser.add_argument("--images", "-i", default="./clean_data/image",
        help="path to images directory")
parser.add_argument("--annotations", "-a", default="./clean_data/annotation",
        help="path to annotations directory")
parser.add_argument("--shape", "-s", default=32, help="shape of input")
parser.add_argument("--prefix", "-p", default="patch_cnn", help="model prefix")
args = parser.parse_args()

shape = int(args.shape)

_train(images_path=args.images,
       annotations_path=args.annotations,
       prefix=args.prefix,
       size=shape)
