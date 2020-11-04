import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector
import BRIEF
import matplotlib.pyplot as plt

im = cv2.imread('../data/model_chickenbroth.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
rows,cols = im.shape
locs, desc = BRIEF.briefLite(im)

count = np.zeros([36,1])
for angle in range(0,36):
    M = cv2.getRotationMatrix2D((cols/2,rows/2),10*angle,1)
    dst = cv2.warpAffine(im,M,(cols,rows))
    locs_r, desc_r = BRIEF.briefLite(dst)
    matches = BRIEF.briefMatch(desc, desc_r)
    count[angle,:] = matches.shape[0]
    
plt.bar(np.arange(36)*10, count[:,0])
plt.xlabel('Angle')
plt.ylabel('Number of Matches')
plt.savefig('q2_5.png')
plt.show()