# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:15:45 2019

@author: Administrator
"""

import scipy.misc as misc
import submission as sub
import numpy as np
import helper

points = np.load("../data/some_corresp.npz")
pts1 = points['pts1']
pts2 = points['pts2']
im1 = misc.imread('../data/im1.png')
im2 = misc.imread('../data/im2.png')

F = sub.eightpoint(pts1,pts2,640)
helper.displayEpipolarF(im1,im2,F)
