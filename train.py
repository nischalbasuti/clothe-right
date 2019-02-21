#!/usr/bin/env python3

import argparse
from patched_cnn import _train
  
parser = argparse.ArgumentParser()
parser.add_argument("--shape", default=32, help="shape of input")
args = parser.parse_args()

shape = int(args.shape)

_train(size=shape)
