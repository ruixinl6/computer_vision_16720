# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 21:11:18 2019

@author: Administrator
"""
import keypointDetect
import numpy as np
import cv2

if __name__ == '__main__':
    path_img = "../data/chickenbroth_01.jpg"
    image = cv2.imread(path_img)
    
    pyramid = keypointDetect.createGaussianPyramid(image, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4])
    #keypointDetect.displayPyramid(pyramid)
    DoG,glevel = keypointDetect.createDoGPyramid(pyramid, levels=[-1,0,1,2,3,4])
    #keypointDetect.displayPyramid(DoG)
    
    curve = keypointDetect.computePrincipalCurvature(DoG)
    
    locsDoG = keypointDetect.getLocalExtrema(DoG, glevel, curve,
        th_contrast=0.03, th_r=12)
    #keypointDetect.displayPyramid(pyramid)