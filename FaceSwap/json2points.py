# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 17:19:08 2018

@author: yiyuezhuo
"""

import json
import numpy as np
import os

def json2points(fname, verbose = True, detect=False):
    with open(fname) as f:
        obj = json.load(f)
    arr = np.array(obj, dtype=np.int)
    
    _fname, _ = os.path.splitext(fname)
    if detect:
        if os.path.exists(_fname+'.jpg'):
            _fname = _fname+'.jpg'
        elif os.path.exists(_fname+'.png'):
            _fname = _fname+'.png'
    np.savetxt(_fname+'.txt', arr, fmt='%i')
    
    
    if verbose:
        print('{} => {}'.format(fname, _fname+'.txt'))
    

import argparse

parser = argparse.ArgumentParser('make points for faceSwapCLI.py')
parser.add_argument('fpath', nargs='+')
parser.add_argument('--model', default='models/shape_predictor_68_face_landmarks.dat')
parser.add_argument('--detect', action='store_true', help="turn off suffix detection")
args = parser.parse_args()


for fname in args.fpath:
    json2points(fname, detect = args.detect)