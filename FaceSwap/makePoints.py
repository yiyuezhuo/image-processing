# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 23:06:07 2018

@author: yiyuezhuo
"""

import sys
import os
import dlib

import numpy as np



def make_point(fname, verbose=True):
    img = dlib.load_rgb_image(fname)
    
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    
    assert len(dets) == 1
    
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        break
    
    parts = np.empty([shape.num_parts, 2], dtype=np.int)
    for i,p in enumerate(shape.parts()):
        parts[i,0] = p.x
        parts[i,1] = p.y
    
    np.savetxt(fname+'.txt', parts, fmt='%i')
    
    if verbose:
        print('{} => {}'.format(fname, fname+'.txt'))


import argparse



parser = argparse.ArgumentParser('make points for faceSwapCLI.py')
parser.add_argument('fpath', nargs='+')
parser.add_argument('--model', default='models/shape_predictor_68_face_landmarks.dat')
args = parser.parse_args()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.model)


for fname in args.fpath:
    make_point(fname)