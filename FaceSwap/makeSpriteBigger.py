# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 21:34:31 2018

@author: yiyuezhuo
"""

from PIL import Image
import numpy as np
import argparse
from shutil import copyfile

def make_bigger_image(fname, times = 8, backup=True):
    im = Image.open(fname)
    if backup:
        #im.save(fname+'.bak')
        im.save(fname)
        copyfile(fname, fname+'.bak')
        
    s = im.size
    im2=im.resize((s[0]*times,s[1]*times))
    im.close()
    im2.save(fname)
    
def make_bigger_points(fname, times = 8, backup=True):
    arr = np.loadtxt(fname, dtype=np.int)
    if backup:
        #np.savetxt(fname + '.bak', arr, fmt='%i')
        copyfile(fname, fname+'.bak')
    arr *= times
    np.savetxt(fname, arr, fmt='%i')

def make_bigger(fname, times = 8, verbose = True, backup=True):
    make_bigger_image(fname, times = times, backup = backup)
    make_bigger_points(fname+'.txt', times = times, backup = backup)
    if verbose:
        print('{} ->(x{}) ->{}'.format(fname,times,fname))
        print('{} ->(x{}) ->{}'.format(fname+'.txt',times,fname+'.txt'))

parser = argparse.ArgumentParser('Make sprite image bigger')
parser.add_argument('fpath', nargs='+')
parser.add_argument('--times', default=8, type=int)
parser.add_argument('--nobackup', action='store_true')

args = parser.parse_args()

for fname in args.fpath:
    make_bigger(fname, times = args.times, backup = not args.nobackup)